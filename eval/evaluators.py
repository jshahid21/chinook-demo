"""
Run the chinook-demo eval suite end-to-end:
  1. Invoke the agent on every row in chinook-demo-eval-v1
  2. Score each response with three evaluators:
       - routing_correct    (deterministic: tool calls match expected order)
       - keywords_present   (deterministic: response contains expected keywords)
       - grounded_response  (LLM-as-judge via openevals: response only states
                             facts present in tool outputs — catches hallucinated
                             track names, prices, invoice IDs)
  3. Print a promote/block verdict and exit 0 (pass) or 1 (fail)

The exit code makes this CI-compatible. In a GitHub Action this same script
blocks merges when the agent regresses.

Docs cheat-sheet (read alongside the code):
  Target function:           https://docs.langchain.com/langsmith/define-target-function
  Evaluator signature:       https://docs.langchain.com/langsmith/code-evaluator-sdk
  LLM-as-judge / openevals:  https://docs.langchain.com/langsmith/openevals
  Reading results:           https://docs.langchain.com/langsmith/read-local-experiment-results
  HITL resume:               https://docs.langchain.com/oss/python/langchain/human-in-the-loop

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
from openevals.llm import create_llm_as_judge

from agent import agent, Context


DATASET_NAME = "chinook-demo-eval-v1"

# Promotion thresholds. If actuals fall below these, the script exits 1 and
# (in CI) blocks the merge. Three-tier rationale:
#   - Routing is "soft": one routing miss in 15 = 93%, still above 90% bar
#   - Keywords are "hard": these rows guard security/decline paths; any miss
#     is a potential data-leak signal, so the bar is 100%
#   - Grounded is "hard": any hallucinated fact is a customer-trust break, so
#     also 100%. Using an LLM-as-judge here because keyword matching can't
#     catch made-up track names — a fluent lie passes the keyword check.
ROUTING_THRESHOLD = 0.90
KEYWORDS_THRESHOLD = 1.00
GROUNDED_THRESHOLD = 1.00


# ============================================================
#   LLM-AS-JUDGE — groundedness check via openevals
# ============================================================
# openevals.create_llm_as_judge(...) returns a callable that takes whatever
# kwargs the prompt references (here: question, tool_outputs, response) and
# returns a feedback dict LangSmith can store as a score.
# Reference-free: we don't compare against a ground-truth answer, we just check
# the response is supported by the tool outputs the agent actually saw.
GROUNDED_PROMPT = """You are evaluating a music store customer support bot.

User question: {question}
Tool outputs the bot saw: {tool_outputs}
Bot's final response: {response}

A grounded response only states facts that appear in the tool outputs (or
honestly says it doesn't have the info). A non-grounded response invents
track names, prices, invoice IDs, or other specifics not in the tool outputs.

Grounding rules:
- COUNTING is grounded: if the bot counts or aggregates rows from tool output
  (e.g. "you have 1 invoice" derived from 10 line-item rows for the same invoice),
  that is grounded reasoning from the data, not fabrication.
- FORMATTING is grounded: if the bot presents tool output as a markdown table or
  bulleted list, that is grounded even if the raw tool output was tuples or JSON.
- REFUND STATUS: if the tool output says "Pending approval", the bot must say
  "pending" or "submitted for review" — saying "approved" is non-grounded.
- INVOICE TOTAL: in purchase history rows, the last field is the INVOICE TOTAL
  shared across all tracks on that invoice (not a per-track price). If all rows
  for an invoice show the same value, the bot stating that as the total IS grounded.

Return true if the response is grounded, false if it invents details.
"""

# claude-haiku-4-5 is small + cheap; fine for grading 15 rows.
# feedback_key becomes the metric name in LangSmith — must match what the
# aggregation loop in main() looks for.
_grounded_judge = create_llm_as_judge(
    prompt=GROUNDED_PROMPT,
    model="anthropic:claude-haiku-4-5",
    feedback_key="grounded_response",
)


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


def grounded_response(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: did the bot only state facts that came from tool outputs?

    LangSmith introspects this signature and passes inputs/outputs/reference_outputs
    by name (order doesn't matter, names do — same convention as the two
    deterministic evaluators above). We extract:
      - the final AIMessage text (what the customer sees)
      - every ToolMessage's content (what the agent actually saw from tools)
    and hand both to the judge, which returns a feedback dict LangSmith stores
    under the 'grounded_response' key.
    """
    messages = (outputs or {}).get("messages") or []

    # Final response = the last AIMessage with non-empty content.
    final_text = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and getattr(msg, "type", "") == "ai":
            final_text = content
            break

    # Tool outputs = every ToolMessage's content, joined for the judge to read.
    # If no tools were called (e.g., a clarifying question), tell the judge
    # explicitly so it doesn't penalize a no-tool response.
    tool_outputs = [
        str(getattr(msg, "content", ""))
        for msg in messages
        if getattr(msg, "type", "") == "tool"
    ]
    tool_outputs_str = "\n---\n".join(tool_outputs) if tool_outputs else "(no tools called)"

    # Kwargs match the {placeholders} in GROUNDED_PROMPT — openevals fills them in.
    return _grounded_judge(
        question=inputs["question"],
        tool_outputs=tool_outputs_str,
        response=final_text,
    )


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
        evaluators=[routing_correct, keywords_present, grounded_response],
        experiment_prefix="chinook-demo-eval",
    )

    # Aggregate per-evaluator scores. evaluate() returns one result per row;
    # each row has one score per evaluator we passed in.
    # Result schema: https://docs.langchain.com/langsmith/read-local-experiment-results
    routing_scores, keyword_scores, grounded_scores = [], [], []
    for r in results:
        for er in r["evaluation_results"]["results"]:
            if er.key == "routing_correct":
                routing_scores.append(er.score)
            elif er.key == "keywords_present":
                keyword_scores.append(er.score)
            elif er.key == "grounded_response":
                grounded_scores.append(er.score)

    routing_pct = sum(routing_scores) / len(routing_scores)
    keyword_pct = sum(keyword_scores) / len(keyword_scores)
    grounded_pct = sum(grounded_scores) / len(grounded_scores)

    routing_ok = routing_pct >= ROUTING_THRESHOLD
    keyword_ok = keyword_pct >= KEYWORDS_THRESHOLD
    grounded_ok = grounded_pct >= GROUNDED_THRESHOLD
    promote = routing_ok and keyword_ok and grounded_ok

    # Print the gate verdict. Per-row failure detail lives in the LangSmith UI
    # under this experiment — no need to re-print it here.
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
    print(f"  Grounded:  {sum(grounded_scores)}/{len(grounded_scores)} "
          f"({grounded_pct:.0%})   threshold ≥ {GROUNDED_THRESHOLD:.0%}   "
          f"{'✓' if grounded_ok else '✗'}")
    print(bar)

    # Exit code 0 = pass, 1 = fail. Same gate CI uses to block PRs.
    sys.exit(0 if promote else 1)


if __name__ == "__main__":
    main()
