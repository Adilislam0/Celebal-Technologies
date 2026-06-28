# Week 8 Quiz — Single Agent Systems & Agent Pipelines
# Adil Islam | A023144825047

---

## Q1. Explain the concept of a stateful directed graph in agent pipelines. How does it differ from a simple sequential chain?

**Stateful Directed Graph:**
A stateful directed graph is a computational structure where:
- **Nodes** are processing units (LLM calls, tool executions, conditional routers, memory reads/writes)
- **Edges** are directed transitions between nodes — they can be unconditional or conditional (based on runtime state)
- **State** is a shared mutable object that flows through the graph and is read/written by each node

Key properties:
- **Cycles are allowed** — a node can loop back to a previous node (e.g., re-try retrieval if confidence is low)
- **Branching** — state at runtime determines which edge to follow (e.g., route to `web_search` vs `database_query` based on query type)
- **Persistence** — the state object accumulates context across all node visits, not just the current step
- Frameworks like **LangGraph** implement this as `StateGraph` with typed state schemas (usually TypedDict)

**How it differs from a simple sequential chain:**

| Dimension | Sequential Chain | Stateful Directed Graph |
|---|---|---|
| Structure | Linear, fixed order | Graph with branches and cycles |
| Control flow | Predetermined | Dynamic, decided at runtime |
| State | Passed forward only | Shared, read+written by any node |
| Looping | Not possible | Supported (retry, reflection loops) |
| Branching | Not possible | Conditional edges on state values |
| Example | Prompt → LLM → Parser | Router → Tool A or B → Evaluator → (loop or end) |

**Example:** A research agent graph:
```
START → query_router → [web_search | doc_retrieval] → summarizer → evaluator
                                                                        ↓
                                                              (if confidence < 0.7)
                                                                        ↓
                                                             → web_search (retry)
```
This is impossible in a sequential chain.

---

## Q2. What is the role of a Tool Node in an agent pipeline, and how are tools registered and invoked?

**Role of a Tool Node:**
A Tool Node is a node in the agent graph that executes external actions the LLM cannot perform natively — file I/O, API calls, database queries, code execution, web search, etc. It acts as the bridge between the LLM's reasoning and the real world.

**How tools are registered:**
Tools are defined as Python functions decorated with `@tool` (LangChain) or described in a JSON schema (OpenAI function calling format). The schema includes:
- `name`: identifier
- `description`: natural language description the LLM uses to decide when to call it
- `parameters`: JSON schema of expected inputs

```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information about the given query."""
    return web_search_api(query)
```

**How tools are invoked:**
1. LLM receives the tool schema alongside the user message
2. LLM outputs a structured `tool_call` object (not plain text): `{"name": "search_web", "args": {"query": "..."}}`
3. The agent runtime intercepts the `tool_call`
4. The Tool Node executes the corresponding Python function
5. Tool output is injected back into the conversation as a `ToolMessage`
6. LLM sees the result and continues reasoning

In LangGraph, a `ToolNode` wraps all registered tools and auto-routes to the right one based on the LLM's `tool_calls` field.

---

## Q3. Describe the concept of an agent loop (think → act → observe). How does this loop terminate?

**The Agent Loop:**

The agent loop is the core runtime cycle of any autonomous agent:

```
Think → Act → Observe → Think → Act → Observe → ... → Final Answer
```

**Think (Reason):**
- The LLM receives the current state: system prompt + conversation history + tool results
- It reasons about what to do next
- Output: either a `tool_call` (needs more info) or a plain text `final_answer`

**Act (Tool execution):**
- If the LLM output contains a `tool_call`, the Tool Node executes it
- Actions include: web search, code execution, database query, file read, API call

**Observe (Tool result injection):**
- The tool result is added to the conversation as a `ToolMessage`
- The LLM sees this result in its next reasoning step
- This grounds the agent's next thought in real, retrieved evidence

**Termination conditions:**
The loop terminates when any of the following is true:
1. **LLM outputs a final answer** — no `tool_calls` in the response, just text
2. **Max iterations reached** — a hard cap (e.g., `max_iterations=10`) prevents infinite loops
3. **Explicit END node** — in LangGraph, a conditional edge routes to `END` when a stopping condition is met in state (e.g., `state["finished"] == True`)
4. **Error threshold** — consecutive tool failures exceed a limit
5. **Human-in-the-loop interrupt** — execution pauses for human approval before proceeding

---

## Q4. What is memory in the context of agents? Differentiate between short-term and long-term memory.

**Memory in agents:**
Memory is the mechanism by which an agent retains and accesses information across reasoning steps, tool calls, or even separate sessions.

**Short-Term Memory (In-Context / Working Memory):**
- Stored directly in the LLM's context window (the prompt)
- Includes: current conversation messages, tool results, intermediate reasoning steps
- Scope: single session / single agent run
- Limit: bounded by context window size (4K–1M tokens depending on model)
- Implementation: the `messages` list in LangGraph state; ChatMessageHistory in LangChain
- Analogy: RAM — fast, temporary, lost when the session ends

**Long-Term Memory (External / Persistent Memory):**
- Stored outside the LLM in a persistent store
- Types:
  - **Episodic**: records of past interactions (vector DB of conversation summaries)
  - **Semantic**: factual knowledge about the user/domain (structured key-value store)
  - **Procedural**: learned workflows or preferences (fine-tuned weights or prompt templates)
- Scope: persists across sessions
- Implementation: vector databases (Chroma, Pinecone), relational DBs, Redis
- Retrieval: semantic similarity search to pull relevant memories into current context
- Analogy: hard disk — slower, but survives restarts

| Aspect | Short-Term | Long-Term |
|---|---|---|
| Location | Context window | External store (DB) |
| Persistence | Session only | Cross-session |
| Capacity | Token-limited | Practically unlimited |
| Access | Implicit (in prompt) | Explicit retrieval call |
| Example | Current chat history | User preferences from last month |

---

## Q5. How does LangGraph enable stateful, multi-step agent workflows? What makes it different from LangChain Expression Language (LCEL)?

**LangGraph:**
LangGraph is a graph-based orchestration library built on top of LangChain for building stateful, cyclical agent workflows.

Core concepts:
- `StateGraph`: the graph object; you define nodes and edges on it
- `State`: a `TypedDict` schema shared across all nodes; nodes read from and write to it
- `Nodes`: Python functions or LangChain runnables; each receives state and returns a state update dict
- `Edges`: unconditional (`graph.add_edge(A, B)`) or conditional (`graph.add_conditional_edges(A, router_fn, {output: node})`)
- `Checkpointing`: built-in state persistence — the graph can be paused, resumed, or replayed at any node
- `Human-in-the-loop`: `interrupt_before` / `interrupt_after` lets you pause for human review mid-graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    tool_results: list
    finished: bool

graph = StateGraph(AgentState)
graph.add_node("llm_call", call_model)
graph.add_node("tool_exec", run_tools)
graph.add_conditional_edges("llm_call", should_continue, {"continue": "tool_exec", "end": END})
graph.add_edge("tool_exec", "llm_call")
graph.set_entry_point("llm_call")
app = graph.compile()
```

**How it differs from LCEL (LangChain Expression Language):**

| Aspect | LCEL | LangGraph |
|---|---|---|
| Structure | Linear pipeline (`chain = A | B | C`) | Directed graph with cycles |
| Control flow | Fixed, sequential | Dynamic, conditional, looping |
| State | Passed forward as return value | Shared mutable TypedDict |
| Cycles | Not supported | Core feature |
| Persistence | None built-in | Checkpointing built-in |
| Use case | Simple prompt chains | Complex multi-step agents |
| Human-in-the-loop | Not supported | Supported via interrupts |

**Summary:** LCEL is great for straightforward pipelines (retrieve → prompt → parse). LangGraph is necessary when you need branching, retrying, looping, or pausing — i.e., real agent behavior.

---

## Q6. What is a multi-agent pipeline? When would you use one instead of a single agent?

**Multi-Agent Pipeline:**
A multi-agent pipeline is a system where multiple specialized agents collaborate to complete a complex task. Each agent has its own LLM instance, tool set, memory, and role. A supervisor/orchestrator agent routes subtasks to the right specialist.

**Architecture patterns:**
- **Supervisor pattern**: one orchestrator LLM decides which worker agent to call next based on the task state
- **Hierarchical**: supervisors can themselves be supervised (nested hierarchies)
- **Sequential pipeline**: Agent A's output feeds Agent B (report drafting → fact-checking → editing)
- **Parallel fan-out**: multiple agents run simultaneously on different sub-problems, results merged

**When to use multi-agent instead of single agent:**

| Reason | Explanation |
|---|---|
| **Context window overflow** | A single agent can't hold all domain knowledge in one context; specialists each carry their own |
| **Specialization** | A coding agent and a research agent have different system prompts, tools, and fine-tuning |
| **Parallelism** | Sub-tasks with no dependencies can run simultaneously (e.g., research 3 topics at once) |
| **Reliability / separation of concerns** | Isolate failures — a failing research agent doesn't crash the writing agent |
| **Task complexity** | Complex workflows with heterogeneous sub-tasks (search + code + analysis + reporting) |

**Rule of thumb:** use a single agent when the task fits in one context window and needs one skill set. Use multi-agent when the task is too long, too broad, or requires parallel execution.

---

## Q7. Explain how you would evaluate an agent pipeline end-to-end. What metrics would you use?

**Evaluation framework for agent pipelines:**

**1. Task Completion Rate:**
Did the agent fully complete the assigned task? Binary or partial-credit grading against a ground-truth expected outcome. Most important top-level metric.

**2. Trajectory / Step Accuracy:**
Was the sequence of tool calls and reasoning steps correct? Evaluate each intermediate step, not just the final answer. Useful for debugging where agents go wrong.

**3. Tool Call Precision:**
- Did the agent call the right tool for each step?
- Were the arguments to each tool call correct?
- Did it avoid unnecessary tool calls?

**4. Final Answer Quality:**
- **Correctness**: factually accurate vs. ground truth (for closed-domain tasks)
- **Faithfulness**: answer grounded in retrieved context, no hallucination (RAGAS faithfulness metric)
- **Relevance**: answer addresses the user's actual question
- **Completeness**: all parts of the question answered

**5. Efficiency Metrics:**
- Number of LLM calls / tool calls per task (lower = more efficient)
- Total latency (wall-clock time to final answer)
- Total tokens consumed (cost proxy)

**6. Safety & Reliability:**
- Error rate: how often does the agent crash, loop infinitely, or produce malformed tool calls?
- Graceful degradation: does it handle tool failures without breaking?

**Evaluation approaches:**
- **Unit tests**: test individual nodes/tools with fixed inputs
- **End-to-end golden datasets**: curated (query, expected_trajectory, expected_answer) triples
- **LLM-as-judge**: use a strong model (GPT-4 / Claude) to score agent responses on correctness, helpfulness, safety
- **Human evaluation**: for open-ended tasks where automatic metrics are insufficient

---

## Summary Table

| Topic | Key Concept |
|---|---|
| Stateful graph | Nodes + shared state + conditional edges + cycles |
| Tool Node | LLM outputs tool_call → Tool Node executes → ToolMessage → LLM continues |
| Agent loop | Think → Act → Observe; terminates on final answer, max iterations, or END node |
| Memory | Short-term = context window; Long-term = external DB with retrieval |
| LangGraph vs LCEL | LangGraph = graph+state+cycles; LCEL = linear sequential chain |
| Multi-agent | Use when: context overflow, specialization needed, parallelism required |
| Evaluation | Task completion, trajectory accuracy, tool precision, answer quality, efficiency |
