from dotenv import load_dotenv
load_dotenv()
# Why: reads .env file and sets ANTHROPIC_API_KEY + LANGSMITH_API_KEY in the
# Python process. Without this, the keys exist in the file but not in memory

from langchain.agents import create_agent
from langchain.tools import tool
import sqlite3

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
        "WHERE g.Name = ? COLLATE NOCASE LIMIT 5",
        (genre,)
    # Why the trailing comma: makes this a 1-item tuple, not just a value in parens.
    # The `?` in the SQL is a placeholder — the database fills it from this tuple.                                                 
    # Doing it this way (called "parameterized SQL") safely escapes user input. 
    # If someone types something sneaky like '; DROP TABLE Customer; --, the
    # database treats it as plain text, not as code to run. Standard defense 
    # against SQL injection. 
    ).fetchall()
    return str(rows)

agent = create_agent(
    model="anthropic:claude-sonnet-4-6", 
    tools=[recommend_tracks],
    # Why this list: the agent only has the abilities you give it here.
    # Without recommend_tracks in this list, the agent could chat about music in general
    # but couldn't actually look anything up in the Chinook database. 
    # This list IS the agent's full toolbox.                                     
    system_prompt="You are a music store assistant. Use the recommend_tracks tool when users ask for music recommendations",
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Recommend some jazz"}]})
    print(result["messages"][-1].content)