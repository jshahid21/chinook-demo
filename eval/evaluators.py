"""
Run the chinook-demo eval suite end-to-end:
  1. Invoke the agent on every row in chinook-demo-eval-v1
  2. Score each response with two evaluators (routing + keyword)
  3. Print a promote/block verdict and exit 0 (pass) or 1 (fail)

The exit code makes this CI-compatible. In a GitHub Action this same script
blocks merges when the agent regresses.

Docs cheat-sheet (read alongside the code):
  Target function:      https://docs.langchain.com/langsmith/define-target-function
  Evaluator signature:  https://docs.langchain.com/langsmith/code-evaluator-sdk
  Reading results:      https://docs.langchain.com/langsmith/read-local-experiment-results
  HITL resume:          https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Run:  python eval/evaluators.py
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from uuid import uuid4

# Make the parent dir importable so we can pull in agent.py from eval/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client
from langgraph.types import Command

from agent import agent, Context


DATASET_NAME = "chinook-demo-eval-v1"

# Promotion thresholds. If actuals fall below these, the script exits 1 and
# (in CI) blocks the merge. Two-tier rationale:
#   - Routing is "soft": one routing miss in 15 = 93%, still above 90% bar
#   - Keywords are "hard": these rows guard security/decline paths; any miss
#     is a potential data-leak signal, so the bar is 100%
ROUTING_THRESHOLD = 0.90
KEYWORDS_THRESHOLD = 1.00


# ============================================================
#   TARGET FUNCTION — what evaluate() runs on every row
# ============================================================

def target(inputs: dict) -> dict:
    """Run the agent for one dataset row. Auto-approves any HITL interrupt
    so refund rows can finish (otherwise they'd hang at the gate forever)."""

    # Fresh thread per row → no checkpointer state bleeds between rows.
    config = {"configurable": {"thread_id": str(uuid4())}}
    ctx = Context(customer_id=inputs["customer_id"])

    state = agent.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        context=ctx,
        config=config,
    )

    # If the run paused at request_refund's HITL gate, auto-approve and
    # resume so the data-layer wall actually runs (the whole point of rows
    # 11/12/14 is to verify what happens AFTER approval).
    if "__interrupt__" in state:
        state = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            context=ctx,
            config=config,
        )

    # evaluate() passes whatever we return here to the evaluators below.
    return {"messages": state["messages"]}


# ============================================================
#   EVALUATORS — score each row's output
# ============================================================

def routing_correct(outputs: dict, reference_outputs: dict) -> bool:
    """Did the agent call the expected tools, in the expected order?

    Returns True/False; LangSmith uses the function name as the metric key.
    If the target errored, outputs may lack 'messages' — treat that as a fail.
    """
    messages = (outputs or {}).get("messages") or []

    # Walk the message list, collect tool names from any AIMessage's tool_calls.
    actual_tools = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []) or []:
            actual_tools.append(tc["name"])

    return actual_tools == reference_outputs["expected_tools"]


def keywords_present(outputs: dict, reference_outputs: dict) -> bool:
    """Does the final response mention at least one expected keyword?

    "any" semantics (not "all") because some rows list alternative phrasings
    (e.g., the vaporwave decline could be 'no', 'not', 'available', 'don't').
    Rows with empty expected_keywords auto-pass — the check doesn't apply.
    """
    expected = reference_outputs["expected_keywords"]
    if not expected:
        return True

    messages = (outputs or {}).get("messages") or []

    # Final response = the last AIMessage with non-empty content.
    final_text = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and getattr(msg, "type", "") == "ai":
            final_text = content.lower()
            break

    return any(kw.lower() in final_text for kw in expected)


# ============================================================
#   MAIN — run eval, aggregate scores, gate on thresholds
# ============================================================

def main():
    client = Client()

    # evaluate() reads the dataset from LangSmith, runs target on each row,
    # scores via the evaluators, and logs everything as one experiment.
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[routing_correct, keywords_present],
        experiment_prefix="chinook-demo-eval",
    )

    # Aggregate per-evaluator scores AND collect failure details so we can
    # print a worklist before the gate verdict.
    # Result schema: https://docs.langchain.com/langsmith/read-local-experiment-results
    routing_scores, keyword_scores = [], []
    failures = []
    for r in results:
        # Score lookup
        routing_pass = keyword_pass = None
        for er in r["evaluation_results"]["results"]:
            if er.key == "routing_correct":
                routing_scores.append(er.score)
                routing_pass = bool(er.score)
            elif er.key == "keywords_present":
                keyword_scores.append(er.score)
                keyword_pass = bool(er.score)

        # If anything failed, capture the actual vs expected for inspection
        if not (routing_pass and keyword_pass):
            run_outputs = r["run"].outputs or {}
            example = r["example"]
            actual_tools = []
            final_text = ""
            for msg in run_outputs.get("messages") or []:
                for tc in getattr(msg, "tool_calls", []) or []:
                    actual_tools.append(tc["name"])
                content = getattr(msg, "content", None)
                if content and getattr(msg, "type", "") == "ai":
                    final_text = content
            failures.append({
                "question": example.inputs.get("question"),
                "customer_id": example.inputs.get("customer_id"),
                "expected_tools": example.outputs.get("expected_tools"),
                "expected_keywords": example.outputs.get("expected_keywords"),
                "actual_tools": actual_tools,
                "final": final_text,
                "routing_pass": routing_pass,
                "keyword_pass": keyword_pass,
            })

    routing_pct = sum(routing_scores) / len(routing_scores)
    keyword_pct = sum(keyword_scores) / len(keyword_scores)

    routing_ok = routing_pct >= ROUTING_THRESHOLD
    keyword_ok = keyword_pct >= KEYWORDS_THRESHOLD
    promote = routing_ok and keyword_ok

    # Print failure detail BEFORE the gate verdict so the worklist is visible.
    if failures:
        print()
        print("FAILURES (worklist):")
        for i, f in enumerate(failures, 1):
            tags = []
            if not f["routing_pass"]:
                tags.append("routing")
            if not f["keyword_pass"]:
                tags.append("keywords")
            print(f"\n  [{i}] {' + '.join(tags)} FAIL  (cust_id={f['customer_id']})")
            print(f"      question:        {f['question']!r}")
            print(f"      expected tools:  {f['expected_tools']}")
            print(f"      actual tools:    {f['actual_tools']}")
            if not f["keyword_pass"]:
                print(f"      expected kws:    {f['expected_keywords']}")
            final = f["final"]
            if len(final) > 200:
                final = final[:200] + "..."
            print(f"      response:        {final!r}")

    # Print the gate verdict.
    bar = "=" * 60
    print()
    print(bar)
    print(f"  {'✅  READY TO PUSH' if promote else '❌  BLOCKED — DO NOT PUSH'}")
    print(f"  Routing:   {sum(routing_scores)}/{len(routing_scores)} "
          f"({routing_pct:.0%})   threshold ≥ {ROUTING_THRESHOLD:.0%}   "
          f"{'✓' if routing_ok else '✗'}")
    print(f"  Keywords:  {sum(keyword_scores)}/{len(keyword_scores)} "
          f"({keyword_pct:.0%})   threshold ≥ {KEYWORDS_THRESHOLD:.0%}   "
          f"{'✓' if keyword_ok else '✗'}")
    print(bar)

    # Exit code 0 = pass, 1 = fail. Same gate CI uses to block PRs.
    sys.exit(0 if promote else 1)


if __name__ == "__main__":
    main()
