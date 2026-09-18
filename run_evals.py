"""
Eval-report runner for the dental agent.

Runs every scenario in eval_scenarios.py (behavioral + safety) against the
REAL Claude model, and writes a scored report -- pass/fail, latency, token
usage, and estimated cost per scenario -- to timestamped JSON and CSV files
under eval_reports/. Unlike test_dental_agent.py (which just asserts
pass/fail for CI), this is meant to be read: open the CSV in a spreadsheet,
or diff two JSON reports to see whether a prompt change made things better
or worse.

This costs real API tokens and takes real time (roughly a few seconds per
scenario turn) -- run it deliberately, not on every save:
    - after changing any system prompt
    - after changing the model (ANTHROPIC_MODEL)
    - before demoing the agent to someone else
    - periodically, to watch for drift if Anthropic updates the model

Usage:
    python run_evals.py                  # run everything
    python run_evals.py --category safety   # only safety/guardrail scenarios
    python run_evals.py --category behavioral
    python run_evals.py --name sunday_request   # a single scenario, by name
"""

import argparse
import csv
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

import dental_agent as agent
from eval_scenarios import ALL_SCENARIOS


def run_scenario(scenario: dict) -> dict:
    """Run one scenario against the real graph and score it. Returns a
    flat dict suitable for writing straight to a CSV row."""
    graph = agent.build_graph()
    thread_id = f"eval-{scenario['name']}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}

    reply = ""
    result = {}
    started_at = time.perf_counter()
    error = None
    try:
        for turn in scenario["turns"]:
            result = graph.invoke(
                {"messages": [HumanMessage(content=turn)], "specialist_hops": 0},
                config=config,
            )
            reply = agent.last_reply_text(result["messages"])
    except Exception as exc:  # noqa: BLE001 -- we want to record any failure, not crash the run
        error = f"{type(exc).__name__}: {exc}"
    latency_seconds = round(time.perf_counter() - started_at, 2)

    usage = result.get("usage", []) if result else []
    cost_usd = agent.estimate_cost_usd(usage)
    total_tokens = sum(u["total_tokens"] for u in usage)

    passed = False
    failure_reason = None
    if error:
        failure_reason = error
    else:
        reply_lower = reply.lower()
        required = scenario["must_contain_any"]
        if required and not any(kw.lower() in reply_lower for kw in required):
            failure_reason = f"missing all of {required}"
        else:
            hit_forbidden = [f for f in scenario["must_not_contain"] if f.lower() in reply_lower]
            if hit_forbidden:
                failure_reason = f"contained forbidden text {hit_forbidden}"
            else:
                passed = True

    return {
        "name": scenario["name"],
        "category": scenario["category"],
        "passed": passed,
        "failure_reason": failure_reason,
        "latency_seconds": latency_seconds,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
        "final_reply": reply,
        "notes": scenario.get("notes", ""),
    }


def run_all(scenarios: list[dict]) -> list[dict]:
    results = []
    for scenario in scenarios:
        print(f"Running: {scenario['name']} ({scenario['category']}) ...", end=" ", flush=True)
        outcome = run_scenario(scenario)
        status = "PASS" if outcome["passed"] else "FAIL"
        print(f"{status}  ({outcome['latency_seconds']}s, {outcome['total_tokens']} tokens)")
        if not outcome["passed"]:
            print(f"    reason: {outcome['failure_reason']}")
        results.append(outcome)
    return results


def write_report(results: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(agent.CLINIC_TIMEZONE).strftime("%Y-%m-%d_%H%M%S")

    json_path = out_dir / f"eval_report_{timestamp}.json"
    csv_path = out_dir / f"eval_report_{timestamp}.csv"

    summary = {
        "timestamp": datetime.now(agent.CLINIC_TIMEZONE).isoformat(),
        "model": agent.MODEL_NAME,
        "total_scenarios": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "total_estimated_cost_usd": round(
            sum(r["estimated_cost_usd"] or 0 for r in results), 4
        ),
        "total_latency_seconds": round(sum(r["latency_seconds"] for r in results), 2),
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = ["name", "category", "passed", "failure_reason", "latency_seconds",
                  "total_tokens", "estimated_cost_usd", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", choices=["behavioral", "safety"], help="only run this category")
    parser.add_argument("--name", help="only run the scenario with this exact name")
    parser.add_argument("--out-dir", default="eval_reports", help="directory to write the report into")
    args = parser.parse_args()

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY before running evals (this needs the real API).", file=sys.stderr)
        sys.exit(1)

    scenarios = ALL_SCENARIOS
    if args.category:
        scenarios = [s for s in scenarios if s["category"] == args.category]
    if args.name:
        scenarios = [s for s in scenarios if s["name"] == args.name]
        if not scenarios:
            print(f"No scenario named '{args.name}'. Available: {[s['name'] for s in ALL_SCENARIOS]}", file=sys.stderr)
            sys.exit(1)

    print(f"Running {len(scenarios)} scenario(s) against model {agent.MODEL_NAME}...\n")
    results = run_all(scenarios)

    json_path, csv_path = write_report(results, Path(args.out_dir))

    passed = sum(1 for r in results if r["passed"])
    total_cost = sum(r["estimated_cost_usd"] or 0 for r in results)
    print(f"\n{passed}/{len(results)} passed. Estimated cost: ${total_cost:.4f}")
    print(f"Report written to:\n  {json_path}\n  {csv_path}")

    if passed < len(results):
        sys.exit(1)  # non-zero exit so this can gate a release if you want it to


if __name__ == "__main__":
    main()
