"""
One-off diagnostic: run a single scenario against the REAL model and print
the raw, unprocessed shape of every message in the final state -- not just
the last one -- so we can see exactly what the model returned instead of
guessing again.

This costs a small amount of real API usage (one scenario's worth of turns).

Usage (from the same folder as dental_agent.py, venv active):
    python debug_content_shape.py sunday_request
    python debug_content_shape.py mixed_intent_price_and_booking
    python debug_content_shape.py reschedule_flow
    python debug_content_shape.py identity_verification_on_cancel

Run it for AT LEAST ONE of the four currently-failing scenarios and paste
the full printed output back -- that tells us definitively what shape the
content is in, instead of us patching extract_text() blind a third time.
"""

import sys
import uuid

from langchain_core.messages import HumanMessage

import dental_agent as agent
from eval_scenarios import ALL_SCENARIOS


def describe_message(i: int, msg) -> None:
    print(f"\n--- message[{i}] ---")
    print(f"type: {type(msg).__name__}")
    print(f"id: {getattr(msg, 'id', None)}")
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        print(f"tool_calls: {tool_calls}")
    content = getattr(msg, "content", msg)
    print(f"content type: {type(content).__name__}")
    print(f"content repr: {content!r}")
    if isinstance(content, list):
        for j, block in enumerate(content):
            print(f"  block[{j}] type: {type(block).__name__}")
            if isinstance(block, dict):
                print(f"  block[{j}] keys: {list(block.keys())}")
            print(f"  block[{j}] repr: {block!r}")
    print(f"extract_text() result: {agent.extract_text(msg)!r}")


def main():
    if len(sys.argv) != 2:
        names = [s["name"] for s in ALL_SCENARIOS]
        print("Usage: python debug_content_shape.py <scenario_name>")
        print(f"Available scenarios: {names}")
        sys.exit(1)

    name = sys.argv[1]
    scenario = next((s for s in ALL_SCENARIOS if s["name"] == name), None)
    if scenario is None:
        print(f"No scenario named '{name}'.")
        sys.exit(1)

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY first.")
        sys.exit(1)

    graph = agent.build_graph()
    thread_id = f"debug-{name}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}

    result = {}
    for turn_num, turn in enumerate(scenario["turns"], start=1):
        print(f"\n{'='*70}\nTURN {turn_num}: {turn!r}\n{'='*70}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=turn)], "specialist_hops": 0},
            config=config,
        )
        print(f"\nFull message list after this turn ({len(result['messages'])} messages):")
        for i, msg in enumerate(result["messages"]):
            describe_message(i, msg)

    print(f"\n{'='*70}")
    print(f"FINAL reply via extract_text(result['messages'][-1]): {agent.extract_text(result['messages'][-1])!r}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
