"""
Travel Assistant Agent
======================
A LangGraph-powered agentic AI that helps users plan travel by combining:
  - Weather lookups (OpenWeatherMap API)
  - Destination search (Tavily Search API)
  - Currency conversion (ExchangeRate-API)
  - A custom date/trip-duration calculator (pure Python)

Run:
    python agent.py           # CLI test
    python app.py             # Gradio UI
"""

import os
import json
import math
from datetime import datetime, date
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ─────────────────────────────────────────────
# 1.  TOOLS
# ─────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather for a given city using the OpenWeatherMap API.

    Args:
        city: Name of the city (e.g. "Paris", "Tokyo", "New York").

    Returns:
        A plain-English weather summary or an error message.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY is not set in .env"

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        desc        = data["weather"][0]["description"].capitalize()
        temp        = data["main"]["temp"]
        feels_like  = data["main"]["feels_like"]
        humidity    = data["main"]["humidity"]
        wind_speed  = data["wind"]["speed"]
        country     = data["sys"]["country"]

        return (
            f"Weather in {city}, {country}:\n"
            f"  Condition : {desc}\n"
            f"  Temp      : {temp}°C (feels like {feels_like}°C)\n"
            f"  Humidity  : {humidity}%\n"
            f"  Wind      : {wind_speed} m/s"
        )
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            return f"City '{city}' not found. Please check the spelling."
        return f"Weather API error: {e}"
    except Exception as e:
        return f"Unexpected error fetching weather: {e}"


@tool
def search_destination(query: str) -> str:
    """
    Search for travel information about a destination using Tavily Search.

    Args:
        query: A natural-language travel query, e.g.
               "top attractions in Barcelona"
               "best time to visit Thailand"
               "visa requirements for Japan"

    Returns:
        A synthesised summary of search results.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in .env"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Tavily may return a synthesised answer directly
        answer = data.get("answer", "")
        results = data.get("results", [])

        parts = []
        if answer:
            parts.append(f"Summary: {answer}")

        for i, r in enumerate(results[:3], 1):
            title   = r.get("title", "")
            content = r.get("content", "")[:300]
            parts.append(f"\n[{i}] {title}\n    {content}…")

        return "\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Search error: {e}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another using ExchangeRate-API.

    Args:
        amount       : The numeric amount to convert (e.g. 100).
        from_currency: ISO 4217 code of the source currency (e.g. "USD").
        to_currency  : ISO 4217 code of the target currency (e.g. "EUR").

    Returns:
        Conversion result as a string, or an error message.
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key:
        return "Error: EXCHANGERATE_API_KEY is not set in .env"

    from_currency = from_currency.upper().strip()
    to_currency   = to_currency.upper().strip()

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}/{amount}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("result") != "success":
            return f"Currency error: {data.get('error-type', 'Unknown error')}"

        rate            = data["conversion_rate"]
        converted       = data["conversion_result"]
        last_updated    = data.get("time_last_update_utc", "N/A")

        return (
            f"Currency Conversion:\n"
            f"  {amount} {from_currency} = {converted:.2f} {to_currency}\n"
            f"  Exchange rate  : 1 {from_currency} = {rate:.4f} {to_currency}\n"
            f"  Rate updated   : {last_updated}"
        )
    except Exception as e:
        return f"Currency conversion error: {e}"


@tool
def calculate_trip_duration(departure_date: str, return_date: str) -> str:
    """
    Calculate the number of nights and days for a trip (custom Python tool).

    Args:
        departure_date : Trip start date in YYYY-MM-DD format (e.g. "2025-08-01").
        return_date    : Trip end date   in YYYY-MM-DD format (e.g. "2025-08-14").

    Returns:
        Trip duration details and a rough per-day budget prompt.
    """
    try:
        dep = datetime.strptime(departure_date, "%Y-%m-%d").date()
        ret = datetime.strptime(return_date,    "%Y-%m-%d").date()

        if ret <= dep:
            return "Error: return_date must be after departure_date."

        delta      = (ret - dep).days
        nights     = delta
        days       = delta + 1          # counting both arrival and departure day
        weeks      = delta / 7
        today      = date.today()
        days_until = (dep - today).days

        return (
            f"Trip Duration Calculator:\n"
            f"  Departure  : {dep.strftime('%A, %B %d %Y')}\n"
            f"  Return     : {ret.strftime('%A, %B %d %Y')}\n"
            f"  Duration   : {nights} nights / {days} days\n"
            f"  ≈ {weeks:.1f} weeks\n"
            f"  Days until departure: {days_until if days_until >= 0 else 'already passed'}"
        )
    except ValueError:
        return "Error: Dates must be in YYYY-MM-DD format (e.g. 2025-08-01)."


# ─────────────────────────────────────────────
# 2.  LANGGRAPH STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Graph state that flows between nodes.

    messages : Full conversation history, automatically merged by add_messages.
                This is the *graph state* — it lives only for the duration of
                one graph invocation unless persisted via checkpointing.
    """
    messages: Annotated[list[AnyMessage], add_messages]


# ─────────────────────────────────────────────
# 3.  NODES
# ─────────────────────────────────────────────

TOOLS = [get_weather, search_destination, convert_currency, calculate_trip_duration]

SYSTEM_PROMPT = """You are an expert AI Travel Assistant.
You help users plan trips by providing:
  • Real-time weather forecasts for any destination
  • Destination search — attractions, culture, visa info, tips
  • Currency conversion between any two currencies
  • Trip duration calculations

Guidelines:
- Always use the appropriate tool when factual data is needed.
- After retrieving data, synthesise it into a clear, friendly response.
- If the user asks about multiple destinations, handle each one.
- For trip planning, proactively suggest relevant follow-up info
  (e.g. after weather → suggest packing tips; after currency → suggest budget).
- Today's date is {today}.
""".format(today=date.today().isoformat())


def build_agent(model_name: str = "gpt-4o-mini") -> tuple:
    """
    Build and compile the LangGraph agent.

    Returns:
        (graph, memory) — the compiled graph and its MemorySaver checkpointer.
    """
    llm = ChatOpenAI(model=model_name, temperature=0.2)
    llm_with_tools = llm.bind_tools(TOOLS)

    # ── Agent node: calls the LLM ──────────────────────────────────────────
    def agent_node(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── Routing: should we call a tool or end? ─────────────────────────────
    def should_continue(state: AgentState) -> str:
        """
        Conditional edge:
          - If the last message contains tool_calls → route to tool_node
          - Otherwise → END (return final answer to user)
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # ── ToolNode: executes whichever tool the LLM chose ───────────────────
    tool_node = ToolNode(TOOLS)

    # ── Build the graph ────────────────────────────────────────────────────
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # After tools finish, always return to the agent for final synthesis
    graph_builder.add_edge("tools", "agent")

    # ── Memory / Checkpointing ─────────────────────────────────────────────
    # MemorySaver stores the full AgentState (including message history)
    # per thread_id. This is different from graph state:
    #   • Graph state  → ephemeral, lives only during one .invoke() call.
    #   • Checkpointed memory → persists across multiple .invoke() calls
    #     as long as the same thread_id is supplied in config.
    memory = MemorySaver()
    graph  = graph_builder.compile(checkpointer=memory)

    return graph, memory


# ─────────────────────────────────────────────
# 4.  CONVENIENCE RUNNER
# ─────────────────────────────────────────────

def run_agent(graph, thread_id: str, user_message: str) -> str:
    """
    Send one user message to the agent and return its final text response.

    The thread_id ties this call to a persisted conversation checkpoint,
    enabling multi-turn memory across separate calls.
    """
    config  = {"configurable": {"thread_id": thread_id}}
    inputs  = {"messages": [HumanMessage(content=user_message)]}
    result  = graph.invoke(inputs, config=config)

    # The last message in state is the agent's final response
    final   = result["messages"][-1]
    return final.content


# ─────────────────────────────────────────────
# 5.  CLI SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    graph, _ = build_agent()
    tid      = "cli-session-001"

    prompts = [
        "What's the weather like in Tokyo right now?",
        "How many days is a trip from 2025-08-10 to 2025-08-24?",
        "Convert 500 USD to JPY for my trip.",
        "What are the top things to do in Tokyo?",
        # Multi-turn: agent should still remember this is about Tokyo
        "Based on everything above, summarise my trip plan.",
    ]

    print("=" * 60)
    print("  Travel Assistant — CLI Demo")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n[User] {prompt}")
        response = run_agent(graph, tid, prompt)
        print(f"[Assistant]\n{response}")
        print("-" * 60)
