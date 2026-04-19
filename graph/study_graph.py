"""LangGraph state, nodes, conditional edges, compiled workflow."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.explanation_agent import explain
from agents.fast_response_agent import quick_answer
from agents.planner_agent import classify_intent
from agents.quiz_agent import generate_quiz
from memory.retriever import retrieve_and_rerank


class AgentState(TypedDict):
    query: str
    intent: str
    context: str
    explanation: str
    quiz: str
    final_output: str
    vector_store: Any
    bm25_store: Any


def planner_node(state: AgentState):
    intent = classify_intent(state["query"])
    return {"intent": intent}


def research_node(state: AgentState):
    context = retrieve_and_rerank(
        state["query"],
        state.get("vector_store"),
        state.get("bm25_store"),
    )
    return {"context": context}


def explanation_node(state: AgentState):
    exp = explain(state["query"], state["context"])
    return {"explanation": "## 📖 Explanation\n" + exp}


def quiz_node(state: AgentState):
    qz = generate_quiz(state["query"], state["context"])
    return {"quiz": "## 📝 Quiz\n" + qz}


def fast_response_node(state: AgentState):
    ans = quick_answer(state["query"])
    return {"final_output": ans}


def synthesizer_node(state: AgentState):
    parts = []
    if state.get("explanation"):
        parts.append(state["explanation"])
    if state.get("quiz"):
        parts.append(state["quiz"])
    output = "\n\n---\n\n".join(parts) if parts else "I could not generate a response."
    return {"final_output": output.strip()}


def route_after_planner(state: AgentState):
    intent = state["intent"]
    if intent in ["quick_question", "unclear_intent", "malicious_intent"]:
        return "fast_path"
    if intent == "learn_only":
        return "explain_only_path"
    if intent == "quiz_only":
        return "quiz_only_path"
    return "learn_and_test_path"


def route_after_research(state: AgentState):
    intent = state["intent"]
    if intent == "learn_only":
        return "explain_only"
    if intent == "quiz_only":
        return "quiz_only"
    return "both"


def route_after_explain(state: AgentState):
    """learn_only → synthesizer; learn_and_test → quiz then synthesizer."""
    if state["intent"] == "learn_only":
        return "synth"
    return "quiz"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("research", research_node)
    workflow.add_node("explain", explanation_node)
    workflow.add_node("quiz", quiz_node)
    workflow.add_node("fast_response", fast_response_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "fast_path": "fast_response",
            "explain_only_path": "research",
            "quiz_only_path": "research",
            "learn_and_test_path": "research",
        },
    )

    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "explain_only": "explain",
            "quiz_only": "quiz",
            "both": "explain",
        },
    )

    # learn_only: explain → synthesizer | learn_and_test: explain → quiz → synthesizer
    workflow.add_conditional_edges(
        "explain",
        route_after_explain,
        {
            "synth": "synthesizer",
            "quiz": "quiz",
        },
    )
    workflow.add_edge("quiz", "synthesizer")
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("fast_response", END)

    return workflow.compile()


study_graph = build_graph()
