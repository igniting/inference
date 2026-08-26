#!/usr/bin/env python3
"""
Novel credit assignment methods for multi-step agent traces.

Two genuinely algorithmic approaches (not prompt engineering):

1. CAST (Causal Ablation via Surrogate Traces):
   For each step, construct a modified trace with that step removed.
   Ask the LLM whether the modified trace could reach the correct answer.
   The step whose removal most improves the outcome is the causal fault.
   This is actual do-calculus intervention — the LLM is a simulator, not a judge.

2. PRSD (Progressive Revelation with Surprise Detection):
   Reveal the trace one step at a time. At each point, ask the LLM to
   predict whether the trace will succeed — WITHOUT showing the final outcome.
   The step that causes the largest drop in predicted success is the fault.
   Each step is a separate API call to eliminate hindsight bias.

Both methods use the LLM as a subroutine/oracle within a genuine algorithm,
rather than asking it to directly attribute blame.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

from external_validation import (
    call_openrouter, parse_json_response, load_traces,
    format_trace_for_llm, CACHE_DIR,
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def format_trace_ablated(trace_data, ablate_step):
    """Format trace with one step replaced by a placeholder."""
    question = trace_data["question"]
    history = trace_data["history"]
    correct_answer = trace_data.get("ground_truth", "unknown")

    lines = [f"TASK: {question}\n"]
    lines.append(f"CORRECT ANSWER: {correct_answer}\n")
    lines.append("AGENT CONVERSATION TRACE (one step has been removed):\n")

    for i, msg in enumerate(history):
        role = msg["role"]
        content = msg["content"]
        if i == ablate_step:
            name = msg.get("name", role)
            lines.append(f"Step {i} [{name}]: [THIS STEP WAS REMOVED — the agent's action here did not occur]\n")
        else:
            if len(content) > 800:
                content = content[:750] + "... [truncated]"
            lines.append(f"Step {i} [{role}]: {content}\n")

    return "\n".join(lines)


def format_trace_prefix(trace_data, up_to_step):
    """Format only the first k steps of the trace (no outcome revealed)."""
    question = trace_data["question"]
    history = trace_data["history"]
    total_steps = len(history)

    lines = [f"TASK: {question}\n"]
    lines.append(f"An agent team is working on this task ({total_steps} steps total).")
    lines.append(f"Here are the first {up_to_step + 1} step(s) so far:\n")

    for i in range(up_to_step + 1):
        msg = history[i]
        role = msg["role"]
        name = msg.get("name", role)
        content = msg["content"]
        if len(content) > 800:
            content = content[:750] + "... [truncated]"
        lines.append(f"Step {i} [{name}]: {content}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CAST: Causal Ablation via Surrogate Traces
# ---------------------------------------------------------------------------

def cast_ablation_prompt(ablated_trace_text, ablate_step, n_steps):
    """Prompt for evaluating one ablation."""
    system = (
        "You evaluate modified agent traces. One step has been removed from the "
        "trace below. Your job: determine whether removing this step would allow "
        "the remaining steps to reach the CORRECT answer.\n\n"
        "Think about information flow: did the removed step introduce wrong "
        "information that later steps relied on? Did it make a bad decision that "
        "constrained later steps? Or was it irrelevant to the failure?\n\n"
        "Respond in JSON:\n"
        "{\"outcome\": \"correctable\" or \"still_wrong\", "
        "\"impact_score\": <float 0.0-1.0 indicating how much removing this step improves things>, "
        "\"reasoning\": \"<one sentence>\"}"
    )
    user = (
        f"{ablated_trace_text}\n\n"
        f"Step {ablate_step} was removed from this {n_steps}-step trace. "
        f"The correct answer is shown above. "
        f"Would removing this step allow the remaining analysis to potentially "
        f"reach the correct answer? Consider:\n"
        f"- Did this step introduce wrong data that downstream steps used?\n"
        f"- Did this step make a decision that locked in the wrong path?\n"
        f"- Or is the error elsewhere and removing this step changes nothing?"
    )
    return system, user


def method_cast(trace_data, model_id):
    """CAST: test each step by ablation."""
    n_steps = len(trace_data["history"])
    step_scores = {}

    for step_i in range(n_steps):
        ablated_text = format_trace_ablated(trace_data, step_i)
        sys_prompt, usr_prompt = cast_ablation_prompt(ablated_text, step_i, n_steps)

        content, latency = call_openrouter(model_id, sys_prompt, usr_prompt)
        parsed = parse_json_response(content)

        if parsed:
            score = parsed.get("impact_score", 0.0)
            outcome = parsed.get("outcome", "still_wrong")
            if outcome == "correctable":
                score = max(score, 0.5)
            step_scores[step_i] = score
        else:
            step_scores[step_i] = 0.0

    if not step_scores:
        return -1, step_scores

    predicted = max(step_scores, key=step_scores.get)
    return predicted, step_scores


# ---------------------------------------------------------------------------
# PRSD: Progressive Revelation with Surprise Detection
# ---------------------------------------------------------------------------

def prsd_prediction_prompt(prefix_text, step_shown, total_steps):
    """Prompt for predicting success after seeing k steps."""
    system = (
        "You are evaluating an agent team's progress on a task. "
        "You will see the task and the steps completed so far. "
        "Based ONLY on what you can see (you do NOT know the final outcome), "
        "estimate the probability that this team will successfully complete the task.\n\n"
        "Consider: Are the agents on the right track? Have they made any mistakes? "
        "Is their approach sound? Are their facts accurate?\n\n"
        "Respond in JSON:\n"
        "{\"probability\": <int 0-100>, \"assessment\": \"<one sentence>\"}"
    )
    user = (
        f"{prefix_text}\n\n"
        f"You have seen {step_shown + 1} of {total_steps} steps. "
        f"Based on what you've observed so far, what is the probability (0-100) "
        f"that this agent team will successfully complete the task with the correct answer? "
        f"Consider the quality of reasoning, accuracy of facts, and soundness of approach."
    )
    return system, user


def method_prsd(trace_data, model_id):
    """PRSD: measure surprise at each step reveal."""
    n_steps = len(trace_data["history"])
    probabilities = []

    for step_k in range(n_steps):
        prefix_text = format_trace_prefix(trace_data, step_k)
        sys_prompt, usr_prompt = prsd_prediction_prompt(prefix_text, step_k, n_steps)

        content, latency = call_openrouter(model_id, sys_prompt, usr_prompt)
        parsed = parse_json_response(content)

        if parsed:
            prob = parsed.get("probability", 50)
            prob = max(0, min(100, int(prob)))
            probabilities.append(prob)
        else:
            probabilities.append(50)

    # Compute surprise (drop in probability) at each step
    surprises = {}
    for i in range(n_steps):
        if i == 0:
            drop = 50 - probabilities[i]
        else:
            drop = probabilities[i - 1] - probabilities[i]
        surprises[i] = drop

    if not surprises:
        return -1, probabilities, surprises

    predicted = max(surprises, key=surprises.get)
    return predicted, probabilities, surprises


# ---------------------------------------------------------------------------
# CAST-Targeted: efficient version using initial screening
# ---------------------------------------------------------------------------

def method_cast_targeted(trace_data, model_id):
    """Efficient CAST: screen candidates first, then ablate top candidates."""
    n_steps = len(trace_data["history"])
    trace_text = format_trace_for_llm(trace_data)

    # Phase 1: Quick screen for top candidates (1 API call)
    screen_system = (
        "You analyze failed agent traces. Identify the 3 most likely steps "
        "where the critical error occurred. Consider factual errors, wrong decisions, "
        "hallucinated data, and unproductive actions.\n\n"
        "Respond in JSON: {\"candidates\": [<int>, <int>, <int>], "
        "\"reasoning\": \"<brief>\"}"
    )
    screen_user = (
        f"{trace_text}\n\n"
        f"This {n_steps}-step trace FAILED. Which 3 steps are most likely to "
        f"contain the critical error? List them in order of likelihood."
    )

    content, _ = call_openrouter(model_id, screen_system, screen_user)
    parsed = parse_json_response(content)

    candidates = []
    if parsed and "candidates" in parsed:
        candidates = [c for c in parsed["candidates"] if 0 <= c < n_steps]

    # Ensure we have at least 3 candidates
    if len(candidates) < 3:
        all_steps = list(range(n_steps))
        for s in all_steps:
            if s not in candidates:
                candidates.append(s)
            if len(candidates) >= 3:
                break

    # Phase 2: Ablation test on candidates (3 API calls)
    step_scores = {}
    for step_i in candidates[:3]:
        ablated_text = format_trace_ablated(trace_data, step_i)
        sys_prompt, usr_prompt = cast_ablation_prompt(ablated_text, step_i, n_steps)

        content, latency = call_openrouter(model_id, sys_prompt, usr_prompt)
        parsed = parse_json_response(content)

        if parsed:
            score = parsed.get("impact_score", 0.0)
            outcome = parsed.get("outcome", "still_wrong")
            if outcome == "correctable":
                score = max(score, 0.5)
            step_scores[step_i] = score
        else:
            step_scores[step_i] = 0.0

    if not step_scores:
        return -1, step_scores, candidates

    predicted = max(step_scores, key=step_scores.get)
    return predicted, step_scores, candidates


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def run_novel_validation(traces, model_id, methods=None):
    """Run novel methods on Who&When traces."""
    if methods is None:
        methods = ["judge", "cast", "cast_targeted", "prsd"]

    results = {m: {"correct": 0, "total": 0, "off_by_one": 0, "errors": [],
                    "details": []} for m in methods}

    for i, trace in enumerate(traces):
        gt_step = trace["mistake_step_int"]
        n_steps = len(trace["history"])
        fname = trace["filename"]
        agent = trace.get("mistake_agent", "?")

        print(f"\n  [{i+1}/{len(traces)}] {fname} (steps={n_steps}, GT={gt_step}, agent={agent})",
              flush=True)

        for method_name in methods:
            try:
                if method_name == "judge":
                    trace_text = format_trace_for_llm(trace)
                    from external_validation import method_judge
                    sys_p, usr_p = method_judge(trace_text, n_steps)
                    content, latency = call_openrouter(model_id, sys_p, usr_p)
                    parsed = parse_json_response(content)
                    pred_step = parsed.get("predicted_step", -1) if parsed else -1
                    detail = {"method": "judge", "pred": pred_step}

                elif method_name == "cast":
                    pred_step, scores = method_cast(trace, model_id)
                    detail = {"method": "cast", "pred": pred_step, "scores": scores}

                elif method_name == "cast_targeted":
                    pred_step, scores, candidates = method_cast_targeted(trace, model_id)
                    detail = {"method": "cast_targeted", "pred": pred_step,
                              "scores": scores, "candidates": candidates}

                elif method_name == "prsd":
                    pred_step, probs, surprises = method_prsd(trace, model_id)
                    detail = {"method": "prsd", "pred": pred_step,
                              "probabilities": probs, "surprises": surprises}
                else:
                    continue

            except Exception as e:
                print(f"    {method_name:16s} ERROR: {e}", flush=True)
                results[method_name]["total"] += 1
                results[method_name]["errors"].append({
                    "trace": fname, "error": str(e)})
                continue

            correct = pred_step == gt_step
            off_by_one = abs(pred_step - gt_step) <= 1
            marker = "+" if correct else ("~" if off_by_one else "x")

            print(f"    {method_name:16s} {marker} pred={pred_step} (GT={gt_step})", flush=True)

            results[method_name]["total"] += 1
            if correct:
                results[method_name]["correct"] += 1
            if off_by_one and not correct:
                results[method_name]["off_by_one"] += 1

            detail["trace"] = fname
            detail["gt"] = gt_step
            detail["correct"] = correct
            detail["off_by_one"] = off_by_one
            results[method_name]["details"].append(detail)

            if not correct:
                results[method_name]["errors"].append({
                    "trace": fname, "pred": pred_step, "gt": gt_step,
                    "agent": agent, "off_by_one": off_by_one,
                })

    return results


def print_report(results, n_traces):
    """Print comparison report."""
    print(f"\n{'='*80}")
    print(f"  NOVEL METHODS VALIDATION: Who&When Benchmark")
    print(f"  {n_traces} traces evaluated")
    print(f"{'='*80}\n")

    print(f"  {'METHOD':<20s} {'Exact':>10s} {'Within±1':>12s} {'Misses':>8s}")
    print(f"  {'-'*54}")

    for method in sorted(results.keys(),
                         key=lambda m: -results[m]["correct"]):
        d = results[method]
        total = d["total"]
        if total == 0:
            continue
        correct = d["correct"]
        obo = d["off_by_one"]
        pct = f"{correct}/{total} ({100*correct/total:.0f}%)"
        pct_obo = f"{correct+obo}/{total} ({100*(correct+obo)/total:.0f}%)"
        print(f"  {method:<20s} {pct:>10s} {pct_obo:>12s} {total-correct-obo:>8d}")

    # SOTA context
    print(f"\n  Reference (state of the art):")
    print(f"  {'FALAT (Jun 2026)':<20s} {'46.0%':>10s}")
    print(f"  {'Who&When baseline':<20s} {'14-20%':>10s}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Novel credit assignment methods: CAST and PRSD")
    parser.add_argument("--max-traces", type=int, default=15)
    parser.add_argument("--model", default="stealth/ox-alpha")
    parser.add_argument("--methods", nargs="+",
                        default=["judge", "cast_targeted", "prsd"])
    parser.add_argument("--key", default="")
    parser.add_argument("--subset", default="Algorithm-Generated")
    args = parser.parse_args()

    import external_validation
    if args.key:
        external_validation.OPENROUTER_API_KEY = args.key
        global OPENROUTER_API_KEY
        OPENROUTER_API_KEY = args.key
    if not external_validation.OPENROUTER_API_KEY:
        print("Error: Set OPENROUTER_API_KEY or pass --key", file=sys.stderr)
        sys.exit(1)

    traces = load_traces("who_and_when_data", args.subset, args.max_traces)
    print(f"\nLoaded {len(traces)} traces from {args.subset}")
    print(f"Running methods: {args.methods}")
    print(f"Model: {args.model}")

    # Estimate API calls
    est = 0
    for m in args.methods:
        if m == "judge":
            est += len(traces)
        elif m == "cast":
            est += len(traces) * 10  # avg steps
        elif m == "cast_targeted":
            est += len(traces) * 4
        elif m == "prsd":
            est += len(traces) * 10
    est_min = est / 8
    print(f"Estimated: ~{est} API calls, ~{est_min:.0f} min")

    results = run_novel_validation(traces, args.model, args.methods)
    print_report(results, len(traces))

    # Save results
    out_dir = "benchmark_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "novel_methods.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
