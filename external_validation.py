#!/usr/bin/env python3
"""
External validation of RPA on the Who&When benchmark (ICML 2025 Spotlight).
184 annotated multi-agent failure traces with ground truth fault steps.

Tests whether RPA's linguistic signals (hedging, conflict, commitment)
generalize from synthetic traces to real-world agent failures.
"""

import json
import glob
import os
import sys
import time
import hashlib
import asyncio
import argparse
from pathlib import Path

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CACHE_DIR = "./benchmark_cache"
RATE_LIMIT_RPM = 8
last_call_time = 0


def call_openrouter(model_id, system_prompt, user_prompt, temperature=0.1):
    """Synchronous OpenRouter API call with caching and rate limiting."""
    global last_call_time
    import urllib.request
    import urllib.error

    cache_key = hashlib.sha256(
        f"{model_id}|{system_prompt}|{user_prompt}|{temperature}|ext_val".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached.get("content"), 0

    elapsed = time.time() - last_call_time
    min_interval = 60.0 / RATE_LIMIT_RPM
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    body = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 3000,
    }).encode()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/inference-benchmark",
    }

    retries = 6
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body, headers=headers, method="POST",
            )
            start = time.time()
            print(f"    [API call attempt {attempt+1}]", flush=True)
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())
            last_call_time = time.time()
            latency_ms = int((last_call_time - start) * 1000)

            content = result["choices"][0]["message"].get("content")
            if content is None:
                wait = min(10 * (attempt + 1), 30)
                print(f"    [null content] retry {attempt+1}, wait {wait}s...", flush=True)
                time.sleep(wait)
                continue

            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"content": content, "model": model_id}, f)

            return content, latency_ms

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(15 * (2 ** attempt), 90)
                print(f"    [429] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    [HTTP {e.code}] retry {attempt+1}...", flush=True)
                time.sleep(5)
        except Exception as e:
            print(f"    [ERR: {e}] retry {attempt+1}...", flush=True)
            time.sleep(5)

    return None, 0


def parse_json_response(text):
    """Parse JSON from LLM response, with fallback brace matching."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def format_trace_for_llm(trace_data):
    """Format a Who&When trace into a readable step-by-step format."""
    question = trace_data["question"]
    history = trace_data["history"]

    lines = [f"TASK: {question}\n"]
    lines.append("AGENT CONVERSATION TRACE (the task was NOT completed correctly):\n")

    for i, msg in enumerate(history):
        role = msg["role"]
        content = msg["content"]
        # Truncate very long steps to keep prompt manageable
        if len(content) > 800:
            content = content[:750] + "... [truncated]"
        lines.append(f"Step {i} [{role}]: {content}\n")

    return "\n".join(lines)


def method_judge(trace_text, n_steps):
    """LLM Judge: direct blame assignment."""
    system = (
        "You are an expert at analyzing failed multi-agent task traces. "
        "Given a conversation trace where agents failed to correctly complete a task, "
        "identify which step contains the critical mistake that caused the failure. "
        "A mistake could be: providing wrong information, making a wrong decision, "
        "hallucinating facts, clicking wrong links, or taking an unproductive action.\n\n"
        "Respond in JSON: {\"predicted_step\": <int>, \"confidence\": <float 0-1>, "
        "\"reasoning\": \"<brief explanation>\"}"
    )
    user = (
        f"{trace_text}\n\n"
        f"This trace has {n_steps} steps (0-indexed). The task FAILED — the agents "
        f"did not arrive at the correct answer. Which single step contains the "
        f"critical mistake that most caused the failure? Consider factual errors, "
        f"wrong tool usage, hallucinated information, and unproductive actions."
    )
    return system, user


def method_rpa(trace_text, n_steps):
    """RPA: Reasoning Pattern Attribution via linguistic signals."""
    system = (
        "You are an expert at analyzing reasoning patterns in agent traces. "
        "For each step in the trace, you will extract three linguistic signals:\n\n"
        "1. HEDGING (H): Uncertainty markers — vague claims, 'should work', "
        "'probably', unverified assumptions, speculative statements, lack of "
        "evidence for claims made.\n"
        "2. CONFLICT (C): Contradiction with available evidence — stating facts "
        "that contradict the task requirements, ignoring search results, "
        "providing information inconsistent with observations, hallucinating data.\n"
        "3. COMMITMENT (K): How much the step constrains the rest of the task — "
        "providing specific data others depend on, making claims that downstream "
        "steps build on, choosing an approach that locks in subsequent actions.\n\n"
        "Respond in JSON with this COMPACT format (only include steps with non-zero signals):\n"
        "{\"steps\": [{\"step\": 0, \"H\": 0.0, \"C\": 0.0, \"K\": 0.0}, ...], "
        "\"predicted_step\": <int>, \"reasoning\": \"<brief>\"}\n\n"
        "IMPORTANT: Keep reasoning under 50 words. Only list steps where H+C+K > 0.\n"
        "The predicted_step should be the step with the highest blame score:\n"
        "blame = (0.3*H + 0.35*C + 0.35*K) * (1 + C*K)"
    )
    user = (
        f"{trace_text}\n\n"
        f"This trace has {n_steps} steps (0-indexed). The task FAILED. "
        f"Analyze each step for hedging (H), conflict (C), and commitment (K) signals. "
        f"Focus especially on steps where agents make factual claims — are those claims "
        f"supported by evidence or search results? Are they consistent with the task "
        f"requirements? Do downstream steps depend on them?"
    )
    return system, user


def compute_rpa_blame(steps_data):
    """Compute RPA blame scores from H, C, K values."""
    scores = {}
    for s in steps_data:
        step = s["step"]
        h = s.get("H", 0)
        c = s.get("C", 0)
        k = s.get("K", 0)
        blame = (0.3 * h + 0.35 * c + 0.35 * k) * (1 + c * k)
        scores[step] = blame
    return scores


def load_traces(data_dir, subset="Algorithm-Generated", max_traces=None):
    """Load Who&When traces."""
    pattern = os.path.join(data_dir, "Who&When", subset, "*.json")
    files = sorted(glob.glob(pattern))
    traces = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        # Parse mistake_step (sometimes string)
        ms = data.get("mistake_step", -1)
        if isinstance(ms, str):
            try:
                ms = int(ms)
            except ValueError:
                continue
        data["mistake_step_int"] = ms
        data["filename"] = os.path.basename(f)
        traces.append(data)

    # Filter to traces with valid mistake steps
    traces = [t for t in traces if 0 <= t["mistake_step_int"] < len(t["history"])]

    if max_traces:
        traces = traces[:max_traces]

    return traces


def run_validation(traces, model_id, methods=None):
    """Run validation on Who&When traces."""
    if methods is None:
        methods = ["judge", "rpa"]

    results = {m: {"correct": 0, "total": 0, "errors": [], "off_by_one": 0} for m in methods}

    for i, trace in enumerate(traces):
        gt_step = trace["mistake_step_int"]
        n_steps = len(trace["history"])
        fname = trace["filename"]
        trace_text = format_trace_for_llm(trace)
        agent = trace.get("mistake_agent", "?")

        print(f"\n  [{i+1}/{len(traces)}] {fname} (steps={n_steps}, GT={gt_step}, agent={agent})", flush=True)

        for method_name in methods:
            if method_name == "judge":
                sys_prompt, usr_prompt = method_judge(trace_text, n_steps)
            elif method_name == "rpa":
                sys_prompt, usr_prompt = method_rpa(trace_text, n_steps)
            else:
                continue

            content, latency = call_openrouter(model_id, sys_prompt, usr_prompt)
            parsed = parse_json_response(content)

            if not parsed:
                print(f"    {method_name:12s} PARSE ERROR", flush=True)
                results[method_name]["total"] += 1
                results[method_name]["errors"].append({
                    "trace": fname, "error": "parse_error",
                })
                continue

            pred_step = parsed.get("predicted_step", -1)

            # For RPA, verify using our own blame formula if steps data available
            if method_name == "rpa" and "steps" in parsed:
                blame_scores = compute_rpa_blame(parsed["steps"])
                if blame_scores:
                    formula_pred = max(blame_scores, key=blame_scores.get)
                    if formula_pred != pred_step:
                        pred_step = formula_pred

            correct = pred_step == gt_step
            off_by_one = abs(pred_step - gt_step) <= 1
            marker = "+" if correct else ("~" if off_by_one else "x")
            reasoning = parsed.get("reasoning", "")[:80]

            print(f"    {method_name:12s} {marker} pred={pred_step} ({latency}ms) {reasoning}", flush=True)

            results[method_name]["total"] += 1
            if correct:
                results[method_name]["correct"] += 1
            if off_by_one and not correct:
                results[method_name]["off_by_one"] += 1
            if not correct:
                results[method_name]["errors"].append({
                    "trace": fname,
                    "pred": pred_step,
                    "gt": gt_step,
                    "agent": agent,
                    "off_by_one": off_by_one,
                    "reasoning": reasoning,
                })

    return results


def print_validation_report(results, subset_name, n_traces):
    """Print validation results."""
    print(f"\n{'='*80}")
    print(f"  EXTERNAL VALIDATION: Who&When Benchmark ({subset_name})")
    print(f"  {n_traces} traces with ground truth fault step annotations")
    print(f"{'='*80}\n")

    print(f"  {'METHOD':<20s} {'Exact':>8s} {'Exact+1':>10s} {'Errors':>8s}")
    print(f"  {'-'*50}")

    for method, data in sorted(results.items(), key=lambda x: -x[1]["correct"]):
        total = data["total"]
        correct = data["correct"]
        obo = data["off_by_one"]
        pct = f"{correct}/{total}={100*correct/total:.0f}%" if total else "N/A"
        pct_obo = f"{correct+obo}/{total}={100*(correct+obo)/total:.0f}%" if total else "N/A"
        print(f"  {method:<20s} {pct:>8s} {pct_obo:>10s} {total-correct-obo:>8d}")

    # Error breakdown
    for method, data in sorted(results.items()):
        errors = data["errors"]
        if not errors:
            continue
        print(f"\n  {method} errors ({len(errors)}):")
        for e in errors[:10]:
            if "error" in e:
                print(f"    {e['trace']:>10s}  {e['error']}")
            else:
                obo = " [off-by-1]" if e.get("off_by_one") else ""
                print(f"    {e['trace']:>10s}  pred={e['pred']} GT={e['gt']} ({e['agent']}){obo}")
                if e.get("reasoning"):
                    print(f"{'':>16s}  {e['reasoning']}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")


def main():
    parser = argparse.ArgumentParser(description="External validation on Who&When benchmark")
    parser.add_argument("--subset", default="Algorithm-Generated",
                       choices=["Algorithm-Generated", "Hand-Crafted", "both"])
    parser.add_argument("--max-traces", type=int, default=30,
                       help="Max traces to evaluate (default: 30)")
    parser.add_argument("--model", default="stealth/ox-alpha")
    parser.add_argument("--methods", nargs="+", default=["judge", "rpa"])
    parser.add_argument("--key", default="")
    args = parser.parse_args()

    global OPENROUTER_API_KEY
    if args.key:
        OPENROUTER_API_KEY = args.key
    if not OPENROUTER_API_KEY:
        print("Error: Set OPENROUTER_API_KEY or pass --key", file=sys.stderr)
        sys.exit(1)

    data_dir = "who_and_when_data"
    if not os.path.exists(data_dir):
        print(f"Error: Dataset not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    subsets = (
        ["Algorithm-Generated", "Hand-Crafted"]
        if args.subset == "both"
        else [args.subset]
    )

    for subset in subsets:
        traces = load_traces(data_dir, subset, args.max_traces)
        print(f"\nLoaded {len(traces)} traces from {subset}")
        print(f"Running methods: {args.methods}")
        print(f"Model: {args.model}")
        est_calls = len(traces) * len(args.methods)
        est_min = est_calls / RATE_LIMIT_RPM
        print(f"Estimated: {est_calls} API calls, ~{est_min:.0f} min")

        results = run_validation(traces, args.model, args.methods)
        print_validation_report(results, subset, len(traces))

    # Save results
    out_dir = "benchmark_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "external_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
