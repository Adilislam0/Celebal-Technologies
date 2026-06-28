"""
Week 8 Assignment: Single Agent Pipeline
=========================================
A stateful directed graph agent that answers research questions by:
  1. Routing the query (needs web search? needs reasoning only?)
  2. Searching a local knowledge base (simulated tool)
  3. Performing web-style keyword expansion (simulated tool)
  4. Synthesising a final grounded answer via Anthropic API

Architecture:
  START → router → [kb_search | reason_only] → synthesiser → evaluator → [END | retry]

State: TypedDict with messages, retrieved_docs, answer, confidence, iterations
Tools: kb_search_tool, keyword_expand_tool, answer_quality_tool
"""

import os
import re
import json
from typing import TypedDict, List, Optional, Literal, Annotated
from dataclasses import dataclass, field

# ── Grok client ──────────────────────────────────────────────────────────
from openai import OpenAI
_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)
LLM_AVAILABLE = bool(os.environ.get("GROQ_API_KEY", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION  (the shared mutable object that flows through the graph)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    The state shared across all graph nodes.
    Each node receives the full state and returns a dict of updates to apply.
    """
    query: str                        # original user question
    messages: List[dict]              # conversation history (role, content)
    retrieved_docs: List[str]         # docs gathered by tool nodes
    keywords: List[str]               # expanded keywords from router
    answer: Optional[str]             # final answer (set by synthesiser)
    confidence: float                 # self-assessed confidence 0.0–1.0
    route: str                        # "kb_search" | "reason_only"
    iterations: int                   # loop counter (prevents infinite loops)
    finished: bool                    # termination flag


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  (in-memory; replace with FAISS store from Week 7 if desired)
# ═══════════════════════════════════════════════════════════════════════════════

KB = {
    "rag": """
RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval
with LLM generation. The pipeline: ingest documents → chunk → embed → store in vector DB →
at query time, embed the query, retrieve top-k chunks, inject as context → LLM generates
a grounded answer. Advanced RAG adds hybrid search (BM25 + dense), re-ranking, HyDE, and
contextual compression. RAG reduces hallucination by grounding the LLM in retrieved evidence.
""",
    "lora": """
LoRA (Low-Rank Adaptation) is a PEFT technique that adds trainable low-rank matrices
A (d×r) and B (r×k) to frozen pretrained weight matrices. Only A and B are updated during
fine-tuning — typically 10,000× fewer parameters than full fine-tuning. At inference, the
adapted weight is W + AB (no extra latency). QLoRA extends LoRA with 4-bit quantization
of the base model, enabling 70B model fine-tuning on a single consumer GPU.
""",
    "react": """
ReAct (Reasoning + Acting) is a prompting paradigm that interleaves chain-of-thought
reasoning with external tool calls. The loop: Thought (reason about what to do) →
Action (call a tool: search, calculator, code) → Observation (tool result) → repeat until
Final Answer. ReAct grounds LLM reasoning in retrieved/computed evidence, preventing
hallucination on factual queries. It is the foundation of modern AI agents.
""",
    "transformer": """
A transformer LLM consists of: Tokenizer (BPE/WordPiece) → Token Embedding + Positional
Encoding → N × Transformer Blocks (Multi-Head Self-Attention + FFN + LayerNorm) →
Linear projection + Softmax over vocabulary. Attention: Q, K, V projections;
score = softmax(QK^T / sqrt(d_k)) * V. KV cache at inference avoids recomputing past
key/value tensors. Emergent capabilities (reasoning, in-context learning) arise from scale.
""",
    "evaluation": """
LLM/RAG evaluation metrics: BLEU and ROUGE measure n-gram overlap (translation/summarization).
BERTScore uses contextual embeddings for semantic similarity. LLM-as-judge uses a strong
model to rate outputs. RAGAS framework provides RAG-specific metrics: Faithfulness
(does the answer follow from retrieved context?), Context Precision (fraction of retrieved
chunks that were useful), Context Recall (fraction of ground-truth evidence retrieved),
and Answer Relevancy (does the answer address the question?).
""",
    "agents": """
An AI agent is an LLM with access to tools and a runtime loop. The agent loop: Think
(LLM reasons, outputs tool_call or final answer) → Act (Tool Node executes the tool) →
Observe (result injected as ToolMessage) → repeat. Termination: LLM outputs final answer,
max iterations reached, or END node triggered. Memory: short-term (context window) and
long-term (external DB). LangGraph implements agents as stateful directed graphs with
TypedDict state, conditional edges, cycles, checkpointing, and human-in-the-loop support.
""",
    "langgraph": """
LangGraph is a graph-based agent orchestration library. Core: StateGraph (the graph),
State (TypedDict shared across nodes), Nodes (functions that read/write state), Edges
(directed; conditional edges use a router function to pick next node), Checkpointing
(built-in state persistence for pause/resume), Human-in-the-loop (interrupt_before/after).
Differs from LCEL (linear sequential chain): LangGraph supports cycles, branching,
shared state, and persistent checkpointing — necessary for real agent behavior.
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS  (Tool Nodes — executed by the agent, results injected into state)
# ═══════════════════════════════════════════════════════════════════════════════

def kb_search_tool(query: str, keywords: List[str]) -> List[str]:
    """
    Tool: Search the local knowledge base.
    Returns: list of relevant document strings.
    In production: replace with FAISS + HybridRetriever from Week 7.
    """
    query_lower = query.lower()
    all_terms = [query_lower] + [k.lower() for k in keywords]
    results = []
    for topic, doc in KB.items():
        # Simple keyword overlap scoring
        score = sum(1 for term in all_terms if term in doc.lower() or topic in query_lower)
        if score > 0:
            results.append((score, doc.strip()))
    results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results[:3]]


def keyword_expand_tool(query: str) -> List[str]:
    """
    Tool: Expand query into keywords for better retrieval coverage.
    In production: use LLM or synonym API.
    """
    # Simple heuristic expansion
    term_map = {
        "rag": ["retrieval", "augmented", "generation", "vector", "embedding"],
        "retrieval": ["rag", "search", "faiss", "bm25", "vector"],
        "lora": ["fine-tuning", "peft", "adapter", "low-rank", "qlora"],
        "fine-tun": ["lora", "peft", "training", "adapter"],
        "agent": ["react", "langgraph", "tool", "loop", "memory"],
        "memory": ["short-term", "long-term", "context", "persistent"],
        "eval": ["ragas", "bleu", "rouge", "bertscore", "faithfulness"],
        "transform": ["attention", "llm", "bert", "gpt", "tokenizer"],
        "graph": ["langgraph", "stateful", "node", "edge", "cycle"],
    }
    keywords = []
    q_lower = query.lower()
    for trigger, expansions in term_map.items():
        if trigger in q_lower:
            keywords.extend(expansions)
    return list(set(keywords))[:8]


def call_llm(messages: List[dict], system: str = "") -> str:
    """Call Groq API (OpenAI-compatible) or fall back to offline synthesiser."""
    if LLM_AVAILABLE and _client:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        response = _client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=600,
            messages=full_messages,
        )
        return response.choices[0].message.content.strip()
    return "[LLM unavailable — set GROQ_API_KEY for full generation]"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES  (each node: AgentState → dict of state updates)
# ═══════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """
    Node 1: Router
    Analyses the query and decides: needs KB search or pure reasoning?
    Also expands query into keywords.
    """
    query = state["query"]
    keywords = keyword_expand_tool(query)

    # Routing heuristic: if query asks for factual info → kb_search
    factual_triggers = [
        "what is", "how does", "explain", "describe", "define",
        "difference between", "compare", "when", "why"
    ]
    needs_retrieval = any(t in query.lower() for t in factual_triggers)
    route = "kb_search" if needs_retrieval else "reason_only"

    print(f"\n[Router] Query: '{query}'")
    print(f"[Router] Keywords expanded: {keywords}")
    print(f"[Router] Route selected: {route}")

    return {
        "keywords": keywords,
        "route": route,
        "messages": state["messages"] + [{
            "role": "assistant",
            "content": f"[Router] Routing to: {route}. Keywords: {keywords}"
        }]
    }


def kb_search_node(state: AgentState) -> dict:
    """
    Node 2a: Knowledge Base Search Tool Node
    Executes the kb_search_tool and injects results into state.
    """
    docs = kb_search_tool(state["query"], state["keywords"])
    print(f"[KB Search] Retrieved {len(docs)} documents.")
    for i, d in enumerate(docs, 1):
        print(f"  Doc {i}: {d[:80]}...")
    return {
        "retrieved_docs": docs,
        "messages": state["messages"] + [{
            "role": "tool",
            "content": f"[KB Search] Retrieved {len(docs)} documents:\n" + "\n---\n".join(docs)
        }]
    }


def reason_only_node(state: AgentState) -> dict:
    """
    Node 2b: Reason-Only Node (no retrieval needed)
    LLM answers from parametric knowledge for simple reasoning tasks.
    """
    print("[Reason-Only] Answering from parametric knowledge.")
    return {
        "retrieved_docs": [],
        "messages": state["messages"] + [{
            "role": "assistant",
            "content": "[Reason-Only] No retrieval needed; using parametric knowledge."
        }]
    }


def synthesiser_node(state: AgentState) -> dict:
    """
    Node 3: Synthesiser
    Builds a grounded prompt from retrieved docs + query and calls the LLM.
    """
    context = "\n\n---\n\n".join(state["retrieved_docs"]) if state["retrieved_docs"] else "No documents retrieved."
    system = (
        "You are a precise, concise technical assistant. "
        "Answer the question using ONLY the provided context. "
        "If context is insufficient, say so. Be specific and structured."
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer (grounded in context, structured, no fluff):"
    )
    messages = [{"role": "user", "content": prompt}]

    answer = call_llm(messages, system=system)

    # If LLM unavailable, synthesise from top doc
    if "[LLM unavailable" in answer and state["retrieved_docs"]:
        best_doc = state["retrieved_docs"][0]
        # Extract most relevant sentences
        query_words = set(state["query"].lower().split())
        sents = re.split(r'(?<=[.!?])\s+', best_doc)
        scored = sorted(sents, key=lambda s: len(query_words & set(s.lower().split())), reverse=True)
        answer = "Based on retrieved context:\n" + " ".join(scored[:4])

    print(f"\n[Synthesiser] Answer generated ({len(answer)} chars)")

    # Simple confidence scoring based on context overlap with query
    query_words = set(state["query"].lower().split())
    answer_words = set(answer.lower().split())
    confidence = min(1.0, len(query_words & answer_words) / max(len(query_words), 1) * 3)
    confidence = round(confidence, 2)

    return {
        "answer": answer,
        "confidence": confidence,
        "messages": state["messages"] + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
    }


def evaluator_node(state: AgentState) -> dict:
    """
    Node 4: Evaluator
    Checks if the answer is satisfactory. If not and iterations < 2, signals retry.
    """
    iterations = state["iterations"] + 1
    confidence = state["confidence"]

    print(f"\n[Evaluator] Iteration {iterations} | Confidence: {confidence}")

    # Termination conditions
    if confidence >= 0.3 or iterations >= 2 or not state["retrieved_docs"]:
        finished = True
        print("[Evaluator] ✅ Answer accepted — terminating.")
    else:
        finished = False
        print("[Evaluator] ⚠️  Low confidence — will retry with expanded search.")

    return {
        "iterations": iterations,
        "finished": finished,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS  (decide which node to visit next)
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_router(state: AgentState) -> Literal["kb_search", "reason_only"]:
    return state["route"]


def route_after_evaluator(state: AgentState) -> Literal["synthesiser", "END"]:
    return "END" if state["finished"] else "synthesiser"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH DEFINITION  (manual implementation — no LangGraph dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleStateGraph:
    """
    Lightweight stateful directed graph implementation.
    Mirrors LangGraph's StateGraph API without requiring the package.
    Nodes: Python functions (state → dict of updates).
    Edges: unconditional or conditional (based on a router function).
    State: shared TypedDict updated via dict merge after each node.
    """

    def __init__(self):
        self.nodes = {}
        self.edges = {}           # node_name → next_node_name (unconditional)
        self.conditional_edges = {}  # node_name → (router_fn, mapping)
        self.entry_point = None

    def add_node(self, name: str, fn):
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, router_fn, mapping: dict):
        self.conditional_edges[from_node] = (router_fn, mapping)

    def set_entry_point(self, name: str):
        self.entry_point = name

    def run(self, initial_state: dict) -> dict:
        """Execute the graph from entry_point until END."""
        state = dict(initial_state)
        current = self.entry_point
        visited = []

        while current and current != "END":
            print(f"\n{'═'*60}")
            print(f"  NODE: {current.upper()}")
            print(f"{'═'*60}")
            visited.append(current)

            # Execute node
            updates = self.nodes[current](state)
            state.update(updates)  # merge updates into state

            # Determine next node
            if current in self.conditional_edges:
                router_fn, mapping = self.conditional_edges[current]
                next_key = router_fn(state)
                current = mapping.get(next_key, "END")
            elif current in self.edges:
                current = self.edges[current]
            else:
                current = "END"

        print(f"\n[Graph] Execution complete. Path: {' → '.join(visited)} → END")
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT PIPELINE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_research_agent() -> SimpleStateGraph:
    """
    Assemble the research agent graph:

    START → router → kb_search ──→ synthesiser → evaluator ──→ END
                  ↘ reason_only ↗                          ↘ synthesiser (retry)
    """
    graph = SimpleStateGraph()

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("kb_search", kb_search_node)
    graph.add_node("reason_only", reason_only_node)
    graph.add_node("synthesiser", synthesiser_node)
    graph.add_node("evaluator", evaluator_node)

    # Set entry point
    graph.set_entry_point("router")

    # Edges
    graph.add_conditional_edges("router", route_after_router, {
        "kb_search": "kb_search",
        "reason_only": "reason_only",
    })
    graph.add_edge("kb_search", "synthesiser")
    graph.add_edge("reason_only", "synthesiser")
    graph.add_edge("synthesiser", "evaluator")
    graph.add_conditional_edges("evaluator", route_after_evaluator, {
        "synthesiser": "synthesiser",  # retry loop
        "END": "END",
    })

    return graph


def run_agent(query: str) -> dict:
    """Run the research agent on a single query."""
    agent = build_research_agent()
    initial_state: AgentState = {
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "retrieved_docs": [],
        "keywords": [],
        "answer": None,
        "confidence": 0.0,
        "route": "kb_search",
        "iterations": 0,
        "finished": False,
    }
    return agent.run(initial_state)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_QUERIES = [
        "What is RAG and how does retrieval work?",
        "Explain LoRA and why it is more efficient than full fine-tuning.",
        "How does the ReAct framework terminate its agent loop?",
        "What metrics does RAGAS use to evaluate RAG systems?",
        "What is the difference between LangGraph and LCEL?",
    ]

    results = []
    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'#'*70}")
        print(f"  QUERY {i}/{len(TEST_QUERIES)}: {q}")
        print(f"{'#'*70}")

        final_state = run_agent(q)

        print(f"\n{'─'*70}")
        print(f"FINAL ANSWER:\n{final_state['answer']}")
        print(f"\nConfidence: {final_state['confidence']} | Iterations: {final_state['iterations']}")
        print(f"Docs retrieved: {len(final_state['retrieved_docs'])}")
        print(f"{'─'*70}")

        results.append({
            "query": q,
            "answer": final_state["answer"],
            "confidence": final_state["confidence"],
            "iterations": final_state["iterations"],
            "docs_retrieved": len(final_state["retrieved_docs"]),
            "route": final_state["route"],
        })

    # Save results
    with open("C:\\Users\\Aadil_islam\\Desktop\\Projects\\Internship\\Celebal\\Celebal-Technologies\\week_8\\agent_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n✅ All {len(TEST_QUERIES)} queries completed.")
    print("Results saved to agent_results.json")
