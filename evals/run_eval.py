"""Model comparison eval for the support chatbot.

Runs every case in cases.json against each selected model through the same tool-calling loop the app uses
(providers.run_turn), scores each conversation with automatic checks and, optionally, an LLM judge, and prints a
comparison table. Full transcripts and scores are saved under evals/results/<timestamp>/.

  python evals/run_eval.py                                     # all available models, all cases
  python evals/run_eval.py --models claude-sonnet-5 gpt-5 --judge claude-opus-5
  python evals/run_eval.py --cases order_ policy_ --workers 8   # only cases whose id starts with these
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

# Load .env (e.g. OPENAI_API_KEY) before providers reads the environment.
load_dotenv(os.path.join(ROOT, ".env"))

import providers  # noqa: E402
from providers import MODELS, STORE_NAME, SYSTEM_PROMPT, ProviderError, list_models, run_turn  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = ["orders", "products", "policies", "escalation"]
RETRY_DELAYS = [5, 15, 30]  # seconds to wait after a rate-limit error before retrying

# A rupee amount in a reply: "Rs 19999", "Rs. 19999", "₹19999", "INR 19999", "19999 rupees", "19999/-"
PRICE_RE = re.compile(r"(?:\brs\.?|₹|\binr)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:rupees|/-)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
EMI_MONTHS = (3, 6, 9, 12, 18, 24)


def _norm(text):
    """Lowercase, drop digit separators ("1,00,000" -> "100000") and unify dashes, so regexes stay simple."""
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text).lower())
    return text.replace("–", "-").replace("—", "-").replace("**", "")


def _numbers(obj):
    """All numbers found anywhere in a JSON-like value."""
    if isinstance(obj, dict):
        return set().union(*map(_numbers, obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*map(_numbers, obj)) if obj else set()
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return {float(obj)}
    return {float(n) for n in NUMBER_RE.findall(_norm(obj))}


def _grounded_amounts(calls, user_turns):
    """Rupee amounts the assistant may state: numbers from tool results or the customer's own messages,
    plus simple derived figures (savings = a - b, EMI = a / months)."""
    base = _numbers([c["output"] for c in calls])
    allowed = base | _numbers(user_turns)
    allowed |= {abs(a - b) for a in base for b in base}
    allowed |= {a / m for a in base for m in EMI_MONTHS}
    return allowed


def check_case(case, transcript, calls):
    """Automatic checks. Returns a list of {"check", "passed", "detail"}."""
    reply = _norm(" ".join(t["text"] for t in transcript if t["role"] == "assistant"))
    final = _norm(transcript[-1]["text"])
    called = [c["name"] for c in calls]
    results = []

    def add(check, passed, detail=""):
        results.append({"check": check, "passed": bool(passed), "detail": detail})

    for name in case.get("expect_tools", []):
        add(f"calls {name}", name in called, f"called: {', '.join(called) or 'no tools'}")
    if case.get("must_call_any"):
        names = case["must_call_any"]
        add(f"calls one of {'/'.join(names)}", set(names) & set(called), f"called: {', '.join(called) or 'no tools'}")
    for name, args in case.get("expect_args", {}).items():
        def matches(c):
            return c["name"] == name and all(
                re.sub(r"[\s-]", "", _norm(c["input"].get(k, ""))) == _norm(v) for k, v in args.items())
        got = [c["input"] for c in calls if c["name"] == name]
        add(f"{name} args {args}", any(map(matches, calls)), f"got: {got}")
    for name in case.get("forbid_tools", []):
        add(f"does not call {name}", name not in called, f"called: {', '.join(called) or 'no tools'}")

    # Content checks look at the final reply; forbidden content is checked across every assistant reply.
    for pattern in case.get("must_match", []):
        add(f"reply matches /{pattern}/", re.search(pattern, final), "")
    for pattern in case.get("must_not_match", []):
        m = re.search(pattern, reply)
        add(f"reply avoids /{pattern}/", not m, f"found: {m.group(0)!r}" if m else "")

    if case.get("check_prices", True):
        allowed = _grounded_amounts(calls, [t["text"] for t in transcript if t["role"] == "user"])
        stated = {float(a or b) for a, b in PRICE_RE.findall(reply)}
        ungrounded = sorted(x for x in stated if not any(abs(x - y) <= 1 for y in allowed))
        add("prices come from tool results", not ungrounded,
            f"ungrounded: {', '.join(f'Rs {x:g}' for x in ungrounded)}" if ungrounded else "")
    return results


# ---------- LLM judge ----------

JUDGE_PROMPT = """You are grading replies from the customer support chatbot of {store}, an Indian electronics retailer.
The chatbot was given these instructions, which define correct behaviour and allowed formatting:
<instructions>
{system_prompt}
</instructions>

What a good answer contains (from the test author):
{notes}

Conversation (tool calls the chatbot made are shown with their results, which are the ground truth):
{conversation}

Score the chatbot's replies from 1 (bad) to 5 (excellent) on:
- correctness: every fact matches the tool results; nothing invented; the right action was taken
- helpfulness: fully answers the customer and suggests a next step when needed
- tone: short, friendly, and follows the formatting rules in the instructions

Respond with only a JSON object: {{"correctness": n, "helpfulness": n, "tone": n, "reason": "<one sentence>"}}"""


def _format_conversation(transcript):
    lines = []
    for t in transcript:
        if t["role"] == "user":
            lines.append(f"CUSTOMER: {t['text']}")
            continue
        for s in (s for round_ in t.get("steps", []) for s in round_):
            lines.append(f"TOOL CALL {s['name']}({json.dumps(s['input'])}) -> {json.dumps(s['output'])}")
        lines.append(f"CHATBOT: {t['text']}")
    return "\n".join(lines)


def complete(model_id, prompt):
    """One plain (no tools) completion from any configured model."""
    cfg = MODELS[model_id]
    if cfg["provider"] == "bedrock":
        extra = {"output_config": {"effort": cfg["effort"]}} if cfg["effort"] else {}
        response = providers._bedrock().messages.create(
            model=cfg["api_model"], max_tokens=4000, messages=[{"role": "user", "content": prompt}], **extra)
        return "".join(b.text for b in response.content if b.type == "text")
    if cfg["provider"] == "bedrock_converse":
        response = providers._converse().converse(
            modelId=cfg["api_model"], messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2000})
        return "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    response = providers._openai().chat.completions.create(
        model=cfg["api_model"], messages=[{"role": "user", "content": prompt}], max_completion_tokens=4000)
    return response.choices[0].message.content or ""


def judge(judge_model, case, transcript):
    prompt = JUDGE_PROMPT.format(store=STORE_NAME, system_prompt=SYSTEM_PROMPT, notes=case.get("notes", "(none)"),
                                 conversation=_format_conversation(transcript))
    text = complete(judge_model, prompt)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge did not return JSON: {text[:200]!r}")
    scores = json.loads(match.group(0))
    parts = [float(scores[k]) for k in ("correctness", "helpfulness", "tone")]
    return {**scores, "overall": round(sum(parts) / len(parts), 2)}


# ---------- Running ----------

def _run_turn_with_retry(model_id, transcript):
    for delay in [*RETRY_DELAYS, None]:
        try:
            return run_turn(model_id, transcript)
        except ProviderError as e:
            if e.status != 429 or delay is None:
                raise
            time.sleep(delay)


def run_case(model_id, case, judge_model=None):
    record = {"model": model_id, "case": case["id"], "category": case.get("category", "other"),
              "passed": False, "checks": [], "judge": None, "error": None, "latency_s": None,
              "tool_calls": 0, "transcript": []}
    transcript, calls = [], []
    started = time.monotonic()
    try:
        for text in case["turns"]:
            transcript.append({"role": "user", "text": text})
            reply, steps = _run_turn_with_retry(model_id, transcript)
            transcript.append({"role": "assistant", "text": reply, "steps": steps})
            calls += [s for round_ in steps for s in round_]
    except Exception as e:  # a failing model should not stop the whole run
        record["error"] = f"{type(e).__name__}: {e}"
        record["transcript"] = transcript
        return record
    record["latency_s"] = round(time.monotonic() - started, 2)
    record["tool_calls"] = len(calls)
    record["transcript"] = transcript
    record["checks"] = check_case(case, transcript, calls)
    record["passed"] = all(c["passed"] for c in record["checks"])
    if judge_model:
        try:
            record["judge"] = judge(judge_model, case, transcript)
        except Exception as e:
            record["judge"] = {"error": f"{type(e).__name__}: {e}"}
    return record


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(records, model_ids):
    rows = []
    for mid in model_ids:
        recs = [r for r in records if r["model"] == mid]
        ok = [r for r in recs if not r["error"]]
        checks = [c for r in ok for c in r["checks"]]
        row = {
            "model": mid,
            "label": MODELS[mid]["label"],
            "cases": len(recs),
            "passed": sum(r["passed"] for r in recs),
            "pass_rate": sum(r["passed"] for r in recs) / len(recs) if recs else None,
            "check_rate": sum(c["passed"] for c in checks) / len(checks) if checks else None,
            "judge": _mean([(r["judge"] or {}).get("overall") for r in ok]),
            "avg_latency_s": _mean([r["latency_s"] for r in ok]),
            "avg_tool_calls": _mean([r["tool_calls"] for r in ok]),
            "errors": sum(bool(r["error"]) for r in recs),
        }
        for cat in CATEGORIES:
            in_cat = [r for r in recs if r["category"] == cat]
            row[cat] = sum(r["passed"] for r in in_cat) / len(in_cat) if in_cat else None
        rows.append(row)
    # Best first: pass rate, then judge score, then speed.
    rows.sort(key=lambda r: (-(r["pass_rate"] or 0), -(r["judge"] or 0), r["avg_latency_s"] or 1e9))
    return rows


def _pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def print_table(rows, with_judge):
    headers = ["Model", "Pass", "Checks", *[c.capitalize() for c in CATEGORIES]]
    headers += ["Judge", "Latency", "Tools", "Errors"] if with_judge else ["Latency", "Tools", "Errors"]
    table = []
    for r in rows:
        line = [r["label"], f"{r['passed']}/{r['cases']} ({_pct(r['pass_rate'])})", _pct(r["check_rate"]),
                *[_pct(r[c]) for c in CATEGORIES]]
        if with_judge:
            line.append("-" if r["judge"] is None else f"{r['judge']:.2f}/5")
        line += ["-" if r["avg_latency_s"] is None else f"{r['avg_latency_s']:.1f}s",
                 "-" if r["avg_tool_calls"] is None else f"{r['avg_tool_calls']:.1f}", str(r["errors"])]
        table.append(line)
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *table)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for line in table:
        print(fmt.format(*line))


def print_failures(records):
    failed = [r for r in records if not r["passed"]]
    if not failed:
        return
    print("\nFailures:")
    for r in sorted(failed, key=lambda r: (r["model"], r["case"])):
        if r["error"]:
            print(f"  {r['model']} / {r['case']}: ERROR {r['error']}")
            continue
        for c in r["checks"]:
            if not c["passed"]:
                print(f"  {r['model']} / {r['case']}: {c['check']}" + (f"  ({c['detail']})" if c["detail"] else ""))


def main():
    parser = argparse.ArgumentParser(description="Compare chatbot models on the eval cases.")
    parser.add_argument("--models", nargs="+", help="model ids from providers.MODELS (default: all available)")
    parser.add_argument("--cases", nargs="+", help="only run cases whose id starts with one of these prefixes")
    parser.add_argument("--judge", help="model id to use as LLM judge, e.g. claude-opus-5 (default: no judge)")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    parser.add_argument("--cases-file", default=os.path.join(EVAL_DIR, "cases.json"))
    parser.add_argument("--out", default=os.path.join(EVAL_DIR, "results"), help="folder for results")
    args = parser.parse_args()

    available = {m["id"] for m in list_models() if m["available"]}
    model_ids = args.models or [mid for mid in MODELS if mid in available]
    for mid in [*model_ids, *([args.judge] if args.judge else [])]:
        if mid not in MODELS:
            parser.error(f"unknown model {mid}. Choose from: {', '.join(MODELS)}")
    skipped = [mid for mid in model_ids if mid not in available]
    if skipped:
        print(f"Skipping {', '.join(skipped)}: OPENAI_API_KEY is not set.")
        model_ids = [mid for mid in model_ids if mid in available]
    if args.judge and args.judge not in available:
        parser.error(f"judge model {args.judge} is not available: OPENAI_API_KEY is not set.")

    with open(args.cases_file) as f:
        cases = json.load(f)
    if args.cases:
        cases = [c for c in cases if c["id"].startswith(tuple(args.cases))]
    if not cases or not model_ids:
        parser.error("nothing to run: no matching cases or models")

    jobs = [(mid, case) for mid in model_ids for case in cases]
    print(f"Running {len(cases)} cases x {len(model_ids)} models = {len(jobs)} conversations"
          + (f", judged by {args.judge}" if args.judge else "") + "\n")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, mid, case, args.judge) for mid, case in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            records.append(r)
            status = "ERROR" if r["error"] else ("PASS " if r["passed"] else "FAIL ")
            timing = f"{r['latency_s']:.1f}s" if r["latency_s"] is not None else ""
            print(f"[{i}/{len(jobs)}] {status} {r['model']:<18} {r['case']:<26} {timing}")

    rows = summarize(records, model_ids)
    print()
    print_table(rows, bool(args.judge))
    print_failures(records)

    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"models": model_ids, "judge": args.judge, "summary": rows,
                   "results": sorted(records, key=lambda r: (r["model"], r["case"]))}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out_dir}/results.json and summary.csv")


if __name__ == "__main__":
    main()
