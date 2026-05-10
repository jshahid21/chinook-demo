from dotenv import load_dotenv
load_dotenv()
# Why: reads .env file and sets ANTHROPIC_API_KEY + LANGSMITH_API_KEY in the
# Python process. Without this, the keys exist in the file but not in memory

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import sqlite3
import os
import sys

# Why a Context class: defines the SHAPE of trusted data the runtime will inject —
# things the model is NOT allowed to control (like which customer is asking).
# customer_id comes from authenticated session at the API boundary, NEVER from chat.
# Per LangChain v1 docs, @dataclass is the canonical pattern (vs Pydantic).

# Use InMemorySaver everywhere EXCEPT under `langgraph dev` (which provides its own).
# langgraph dev loads `langgraph_api` into sys.modules before importing this file;
# we use that as the runner-detection signal. Keeps HITL working for: python agent.py,
# python eval/evaluators.py, pytest, etc.
checkpointer = None if "langgraph_api" in sys.modules else InMemorySaver()

@dataclass 
class Context:
    customer_id: int

@tool
def recommend_tracks(genre: str) -> str:
    """Recommend tracks in a given genre. Use when users ask for music in a specific genre."""
    # Why mode=ro: connection layer read-only enforcement. Even if a future tool attempts INSERT/UPDATE/DELETE, 
    # sqlite refuses the connnection
    conn = sqlite3.connect("file:Chinook.db?mode=ro", uri=True)
    # Why 3 joins: a Track only stores its genre ID and album ID — it doesn't have a direct
    # link to the artist. To get the artist name, we go: Track → Album → Artist.
    # Joining Genre lets us filter by genre NAME (like "Jazz") instead of by ID number.
    rows = conn.execute(
        "SELECT t.Name, ar.Name FROM Track t "
        "JOIN Genre g ON t.GenreId = g.GenreId "
        "JOIN Album al ON t.AlbumId = al.AlbumId "
        "JOIN Artist ar ON al.ArtistId = ar.ArtistId "
        "WHERE g.Name = ? COLLATE NOCASE "
        "ORDER BY RANDOM() LIMIT 5",
        (genre,)
    # Why the trailing comma: makes this a 1-item tuple, not just a value in parens.
    # The `?` in the SQL is a placeholder — the database fills it from this tuple.                                                 
    # Doing it this way (called "parameterized SQL") safely escapes user input. 
    # If someone types something sneaky like '; DROP TABLE Customer; --, the
    # database treats it as plain text, not as code to run. Standard defense 
    # against SQL injection. 
    ).fetchall()
    return str(rows)

@tool
def get_my_recent_purchases(runtime: ToolRuntime[Context]) -> str:
    """Get the authenticated customer's recent purchases (their own, never another customer's).
    Use this when the user asks about their own purchase history, recent orders, or what they bought.
    Customer identity is read from the trusted runtime context — never from chat input."""
    # Why runtime.context.customer_id and NOT a parameter:
    # if customer_id were a function arg, the model could pass any value (including another
    # customer's). By reading from runtime.context, the customer is locked to whoever the
    # API boundary authenticated. Even if the user types "show me Bob's purchases," the 
    # model can't override this.
    customer_id = runtime.context.customer_id
    conn = sqlite3.connect("file:Chinook.db?mode=ro", uri=True)                                                   
    rows = conn.execute(
        "SELECT i.InvoiceId, t.Name, ar.Name, i.InvoiceDate "
        "FROM Invoice i "
        "JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId "
        "JOIN Track t ON il.TrackId = t.TrackId "
        "JOIN Album al ON t.AlbumId = al.AlbumId "
        "JOIN Artist ar ON al.ArtistId = ar.ArtistId "
        "WHERE i.CustomerId = ? "                       # locked to authenticated customer
        "ORDER BY i.InvoiceDate DESC LIMIT 10",
        (customer_id,)
    ).fetchall()
    return str(rows)

@tool
def request_refund(invoice_id: int, runtime: ToolRuntime[Context]) -> str:
    """Request a refund for a specific invoice belonging to the authenticated customer. 
    Use this when the user asks for a refund and has identified a specific invoice or purchase.
    Customer identity comes from runtime context - never from chat."""
    customer_id = runtime.context.customer_id
    conn = sqlite3.connect("file:Chinook.db?mode=ro", uri=True)
        # Verify the invoice exists AND belongs to this customer
        # (data-layer wall: even if customer_id is wrong, this SELECT returns 0 rows)
    row = conn.execute(
        "SELECT InvoiceId, Total FROM Invoice "
        "WHERE InvoiceId = ? AND CustomerId = ?",
        (invoice_id, customer_id)
    ).fetchone()

    if row is None:
        return f"Invoice {invoice_id} not found for this customer."

    # No real DB write - agent returns confirmation only. HITL middleware pauses this call before reaching here.
    # Select above is the third wall (rejects if the invoice doesn't belong to authenticated customer)
    return f"Refund of ${row[1]:.2f} requested for invoice {row[0]}. Pending approval."

agent = create_agent(
    model="anthropic:claude-sonnet-4-6", 
    # Why this list: the agent only has the abilities you give it here.
    # Without recommend_tracks in this list, the agent could chat about music in general
    # but couldn't actually look anything up in the Chinook database. 
    # This list IS the agent's full toolbox. 
    tools=[recommend_tracks, get_my_recent_purchases, request_refund],
    # Why context_schema: registers the Context class so the runtime knows what
    # shape to expect when invoke is called with context=. Without this, passing
    # context= to invoke is silently ignored — tools see runtime.context as None. 
    context_schema=Context, 
    middleware=[
        # Why redact (not block): customer support routinely sees emails ("refund to my account
        # at alice@example.com"). Blocking would refuse the message; redacting lets the agent
        # still help while keeping the email out of the model's context and out of LangSmith
        # traces. Compliance baseline for any agent handling customer support.
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        HumanInTheLoopMiddleware(
            interrupt_on={"request_refund": True},
            description_prefix="Refund pending approval"
            ),
        # cost runaway protection: limits single user max model calls in a thread and in a turn 
        ModelCallLimitMiddleware(thread_limit=20, run_limit=10),
        ],
    # HITL requires checkpointing to handle interrupts.
    # Use InMemorySaver in dev, AsyncPostgresSaver in prod
    # Must configure a checkpointer to persist the graph state across interrupts.
    checkpointer=checkpointer,
    system_prompt=(
        "You are a music store assistant. "
        "Use the recommend_tracks tool when users ask for music recommendations. "
        "Use get_my_recent_purchases when the user asks about their own purchase history. "
        "Use request_refund when the user asks to refund a specific invoice from their purchase history. "
        "IMPORTANT: never accept customer IDs from chat — customer identity is set by the system."
        ),
    )

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-1"}}
    # First invoke - should pause at HITL gate
    print("\n=== STEP 1: Invoking refund - expect pause ===")
    result = agent.invoke({"messages": [{"role": "user", "content": "Please process a refund for invoice 156. I want this refunded now."}]},
                            # Why context here: in production, your auth layer would set customer_id
                            # from a logged-in session. For demo/smoke-test, I pass it manually as
                            # Customer 14 (Mark Philips in Chinook). When running via `langgraph dev`,
                            # the same value gets set in Studio's Context panel instead.
                            context=Context(customer_id=14),
                            config=config,
                            version="v2",
                            )
    print("Result type:", type(result).__name__)
    if hasattr(result, "interrupts") and result.interrupts:
        print("PAUSED. Interrupts:", result.interrupts)
    else:
        print("Did NOT pause:", result)
    
    # Resume with approve
    print("\n=== STEP 2: Resuming with approve ===")
    final = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        context=Context(customer_id=14),
        config=config,
        version="v2",
    )
    print("Final type:", type(final).__name__)
    print("Agent response:", final.value["messages"][-1].content)