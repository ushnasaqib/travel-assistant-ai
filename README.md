# ✈️ AI Travel Assistant

An agentic AI travel companion built with **LangChain**, **LangGraph**, and **Gradio**. It answers travel questions by reasoning over real-time data from three external APIs and one custom Python tool.

---

## Use Case

Planning a trip involves many moving parts: checking local weather, researching what to do, handling foreign currency, and figuring out how long you'll be away. This assistant bundles all of that into one conversational interface that remembers your session context across multiple questions.

---

## Tools Integrated

| # | Tool | Type | API / Source |
|---|------|------|-------------|
| 1 | `get_weather` | External API | OpenWeatherMap |
| 2 | `search_destination` | External API | Tavily Search |
| 3 | `convert_currency` | External API | ExchangeRate-API |
| 4 | `calculate_trip_duration` | Custom Python | Pure Python (`datetime`) |

---

## APIs Used

| API | Free Tier | Purpose |
|-----|-----------|---------|
| [OpenWeatherMap](https://openweathermap.org/api) | ✅ Yes | Current weather by city |
| [Tavily](https://tavily.com) | ✅ Yes | AI-powered web search |
| [ExchangeRate-API](https://www.exchangerate-api.com) | ✅ Yes | Live currency conversion |
| [OpenAI](https://platform.openai.com) | ❌ Paid | LLM backbone (GPT-4o-mini) |

---

## LangGraph Workflow

```
User Input
   │
   ▼
Gradio UI  ─── thread_id ───►  MemorySaver (checkpoint)
   │
   ▼
LangGraph.invoke()
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│  AgentState { messages: list[AnyMessage] }               │
│                                                          │
│  agent_node ──► should_continue? ──► tool_calls? ──Yes──►│
│       ▲                    │                   ToolNode  │
│       │                    No                     │      │
│       └────────────────────┼───────────────────────┘     │
│                            │                             │
│                           END                            │
└──────────────────────────────────────────────────────────┘
   │
   ▼
Final response → Gradio → User
```

### Nodes

| Node | Role |
|------|------|
| `agent_node` | Calls GPT-4o-mini with all tools bound. Decides to answer or call a tool. |
| `ToolNode` | Executes the tool chosen by the LLM and returns a `ToolMessage`. |

### Edges

| Edge | Type | Description |
|------|------|-------------|
| `START → agent` | Fixed | Entry point |
| `agent → should_continue` | Conditional | Routes based on whether tool_calls exist |
| `should_continue → tools` | Conditional | LLM wants to call a tool |
| `should_continue → END` | Conditional | LLM has a final answer |
| `tools → agent` | Fixed | Tool result fed back to agent for synthesis |

---

## Memory Implementation

### Graph State vs. Memory

| Aspect | Graph State | Checkpointed Memory |
|--------|-------------|---------------------|
| Scope | Single `.invoke()` call | Persists across multiple calls |
| Storage | In-memory Python dict | `MemorySaver` (or Redis/SQL in production) |
| Key | N/A | `thread_id` |
| Purpose | Pass data between nodes | Recall prior conversation turns |

### How it works

1. Every Gradio session generates a unique `thread_id` (UUID).
2. Each call to `graph.invoke()` passes `config = {"configurable": {"thread_id": tid}}`.
3. `MemorySaver` serialises and restores `AgentState.messages` between calls.
4. The agent sees full conversation history, enabling multi-turn reasoning.

### Benefit

Without memory, each question is isolated. With memory, a user can ask:
> "What's the weather in Tokyo?" → "Convert 500 USD to JPY" → **"Summarise my trip plan."**
…and the agent connects all three turns into a coherent answer.

---

## How to Run

### 1. Clone / download

```bash
git clone https://github.com/your-username/travel-assistant
cd travel-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

```bash
cp .env.example .env
# Open .env and paste your API keys
```

### 4. Run the Gradio app

```bash
python app.py
# → Open http://127.0.0.1:7860 in your browser
```

### 5. (Optional) CLI test

```bash
python agent.py
```

---

## Example Prompts

```
What's the weather like in Bangkok right now?
I'm flying to Japan from 2025-10-01 to 2025-10-15. How long is the trip?
Convert 2000 GBP to JPY.
What are the top attractions in Kyoto?
Do I need a visa to visit Thailand?
Based on all the above, give me a summary travel brief for my Japan trip.
```

---

## Challenges Faced

- **Tool routing ambiguity**: The LLM occasionally tried to answer currency questions from training knowledge instead of calling the API. Fixed by strengthening the system prompt to always use tools for factual data.
- **Multi-step memory**: Ensuring `add_messages` correctly merges history without duplication required understanding LangGraph's reducer pattern.
- **Gradio state management**: Passing `thread_id` as `gr.State` (not a visible component) while keeping it stable across turns needed care.

---

## Future Improvements

- 🗄 **Persistent checkpointing** — swap `MemorySaver` for `PostgresSaver` or `RedisSaver` so sessions survive server restarts.
- 🧳 **Flight search tool** — integrate Amadeus or Skyscanner API.
- 🏨 **Hotel search tool** — integrate Booking.com or Airbnb API.
- 🗺 **Map display** — embed an interactive map using Folium.
- 🤖 **Multi-agent** — add a "budget planner" sub-agent that aggregates costs.
- 📡 **LangSmith tracing** — add observability for debugging tool calls.
- 🚀 **Deploy** — publish to Hugging Face Spaces for public access.

---

## Architecture Diagram

See [`workflow_diagram.svg`](./workflow_diagram.svg)

---

## Project Structure

```
travel-assistant/
├── agent.py              # LangGraph agent, tools, state, graph
├── app.py                # Gradio UI
├── requirements.txt
├── .env.example          # API key template
├── workflow_diagram.svg  # Architecture diagram
└── README.md
```
