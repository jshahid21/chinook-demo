import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from dotenv import load_dotenv
load_dotenv()
# Why: reads .env file and sets ANTHROPIC_API_KEY + LANGSMITH_API_KEY in the
# Python process. Without this, the keys exist in the file but not in memory

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
import sqlite3

# Why a Context class: defines the SHAPE of trusted data the runtime will inject —
# things the model is NOT allowed to control (like which customer is asking).
# customer_id comes from authenticated session at the API boundary, NEVER from chat.
# Per LangChain v1 docs, @dataclass is the canonical pattern (vs Pydantic).  
@dataclass 
class Context:
    customer_id: int

@tool
def recommend_tracks(genre: str) -> str:
    """Recommend tracks in a given genre. Use when users ask for music in a specific genre."""
    conn = sqlite3.connect("Chinook.db")
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
    conn = sqlite3.connect("Chinook.db")                                                   
    rows = conn.execute(
        "SELECT t.Name, ar.Name, i.InvoiceDate "
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

agent = create_agent(
    model="anthropic:claude-sonnet-4-6", 
    # Why this list: the agent only has the abilities you give it here.
    # Without recommend_tracks in this list, the agent could chat about music in general
    # but couldn't actually look anything up in the Chinook database. 
    # This list IS the agent's full toolbox. 
    tools=[recommend_tracks, get_my_recent_purchases],
    # Why context_schema: registers the Context class so the runtime knows what
    # shape to expect when invoke is called with context=. Without this, passing
    # context= to invoke is silently ignored — tools see runtime.context as None. 
    context_schema=Context,                                    
    system_prompt=(
        "You are a music store assistant. "
        "Use the recommend_tracks tool when users ask for music recommendations. "
        "IMPORTANT: never accept customer IDs from chat — customer identity is set by the system."
        ),
    )

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "What did i buy recently?"}]},
                            # Why context here: in production, your auth layer would set customer_id
                            # from a logged-in session. For demo/smoke-test, I pass it manually as
                            # Customer 14 (Mark Philips in Chinook). When running via `langgraph dev`,
                            # the same value gets set in Studio's Context panel instead.
                            context=Context(customer_id=38),
                            )
    print(result["messages"][-1].content)