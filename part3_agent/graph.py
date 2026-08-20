"""
Core LangGraph agent. 5 nodes, 2 conditional-edge decision points.
Multi-turn state (last_order_features) persists via MemorySaver checkpointer,
keyed by thread_id -- same thread_id carries state, a new thread_id starts clean.
"""

from typing import TypedDict, Optional, Dict, Any, List, cast
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from guardrails import (
    check_prompt_injection,
    check_groundedness,
    GROUNDEDNESS_THRESHOLD,
)
from mock_llm import (
    mock_classify_intent,
    mock_compose_policy_answer,
    mock_compose_return_risk_answer,
    mock_compose_image_answer,
    mock_check_injection_response,
)
from embed_index import search as rag_search
from tools.check_return_risk import check_return_risk
from tools.classify_product_image import classify_product_image


class AgentState(TypedDict, total=False):
    user_input: str
    order_features: Optional[Dict[str, Any]]
    image_path: Optional[str]
    injection_blocked: bool
    intent: str
    retrieved_chunks: List[dict]
    groundedness: dict
    tool_output: dict
    final_answer: dict
    last_order_features: Optional[
        Dict[str, Any]
    ]  # persisted across turns via checkpointer


def input_guard_node(state: AgentState) -> dict:
    state_data = cast(Dict[str, Any], state)
    check = check_prompt_injection(state_data["user_input"])
    return {"injection_blocked": check["blocked"]}


def route_after_guard(state: AgentState) -> str:
    # Conditional edge #1: injected input skips straight to response node.
    state_data = cast(Dict[str, Any], state)
    return (
        "response_generation"
        if state_data["injection_blocked"]
        else "intent_classification"
    )


def intent_node(state: AgentState) -> dict:
    state_data = cast(Dict[str, Any], state)
    intent = mock_classify_intent(state_data["user_input"])
    return {"intent": intent}


def route_after_intent(state: AgentState) -> str:
    # Conditional edge #2: branch by classified intent.
    state_data = cast(Dict[str, Any], state)
    if state_data["intent"] == "policy":
        return "rag_retrieval"
    return "tool_calling"


def rag_retrieval_node(state: AgentState) -> dict:
    state_data = cast(Dict[str, Any], state)
    chunks = rag_search(state_data["user_input"], k=3)
    grounded = check_groundedness(chunks, threshold=GROUNDEDNESS_THRESHOLD)
    return {"retrieved_chunks": chunks, "groundedness": grounded}


def tool_calling_node(state: AgentState) -> dict:
    state_data = cast(Dict[str, Any], state)
    if state_data["intent"] == "return_risk":
        # Multi-turn state: use newly-provided order_features, else fall back to the
        # order mentioned in a prior turn (carried via checkpointer as last_order_features).
        order_features = state_data.get("order_features") or state_data.get(
            "last_order_features"
        )
        if order_features is None:
            return {
                "tool_output": {"error": "No order details provided or remembered."}
            }
        result = check_return_risk(order_features)
        # Persist this order for follow-up turns in the same thread_id.
        return {"tool_output": result, "last_order_features": order_features}

    elif state_data["intent"] == "product_category":
        image_path = state_data.get("image_path")
        if image_path is None:
            return {"tool_output": {"error": "No image path provided."}}
        result = classify_product_image(image_path)
        return {"tool_output": result}

    return {"tool_output": {"error": "Unknown intent for tool calling."}}


def response_generation_node(state: AgentState) -> dict:
    state_data = cast(Dict[str, Any], state)
    if state_data.get("injection_blocked"):
        return {"final_answer": mock_check_injection_response()}

    if state_data["intent"] == "policy":
        g = state_data["groundedness"]
        answer = mock_compose_policy_answer(
            state_data["retrieved_chunks"],
            g["grounded"],
            g["top_score"],
            g["threshold"],
        )
        return {"final_answer": answer}

    tool_output = state_data["tool_output"]
    if "error" in tool_output:
        return {
            "final_answer": {
                "answer": tool_output["error"],
                "source": "policy_kb",
                "confidence": 0.0,
            }
        }

    if state_data["intent"] == "return_risk":
        return {"final_answer": mock_compose_return_risk_answer(tool_output)}
    else:
        return {"final_answer": mock_compose_image_answer(tool_output)}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("intent_classification", intent_node)
    builder.add_node("rag_retrieval", rag_retrieval_node)
    builder.add_node("tool_calling", tool_calling_node)
    builder.add_node("response_generation", response_generation_node)

    builder.set_entry_point("input_guard")
    builder.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {
            "response_generation": "response_generation",
            "intent_classification": "intent_classification",
        },
    )
    builder.add_conditional_edges(
        "intent_classification",
        route_after_intent,
        {
            "rag_retrieval": "rag_retrieval",
            "tool_calling": "tool_calling",
        },
    )
    builder.add_edge("rag_retrieval", "response_generation")
    builder.add_edge("tool_calling", "response_generation")
    builder.add_edge("response_generation", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


agent_graph = build_graph()


def run_agent(
    text: str,
    thread_id: str,
    order_features: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
) -> dict:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    input_state: AgentState = {
        "user_input": text,
        "order_features": order_features,
        "image_path": image_path,
    }
    result = agent_graph.invoke(input_state, config=config)
    return result["final_answer"]


if __name__ == "__main__":
    sample_order = {
        "price_inr": 1800,
        "discount_pct": 45.0,
        "customer_tenure_days": 90,
        "num_previous_orders": 2,
        "num_previous_returns": 1,
        "delivery_distance_km": 320.0,
        "delivery_days": 6,
        "is_weekend_order": 1,
        "rating_given": None,
        "product_category": "Apparel",
        "payment_method": "COD",
    }

    print("=== Multi-turn conversation (thread: demo-session) ===")
    print("\nTurn 1: providing full order details")
    r1 = run_agent(
        "Check the return risk for this order.",
        thread_id="demo-session",
        order_features=sample_order,
    )
    print(r1)

    print(
        "\nTurn 2: follow-up referencing 'that order' with NO new order_features passed"
    )
    r2 = run_agent(
        "What's the risk bucket for that order again?",
        thread_id="demo-session",
        order_features=None,
    )
    print(r2)

    print("\n=== Fresh conversation (thread: new-session, no prior turns) ===")
    r3 = run_agent(
        "What's the risk bucket for that order again?",
        thread_id="new-session",
        order_features=None,
    )
    print(r3)
