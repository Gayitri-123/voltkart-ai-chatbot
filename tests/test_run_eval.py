import json
import os

import pytest

import run_eval
from conftest import claude_message, completion


def _transcript(user, reply, calls=()):
    return [{"role": "user", "text": user}, {"role": "assistant", "text": reply, "steps": [list(calls)] if calls else []}]


ORDER_CALL = {"name": "get_order_status", "input": {"order_id": "ORD10002", "phone": "9876543210"},
              "output": {"status": "Out for delivery", "items": [{"price": 19999}]}}


def test_norm():
    assert run_eval._norm("Rs **1,00,000** – ok") == "rs 100000 - ok"


def test_cases_file_is_valid():
    with open(os.path.join(run_eval.EVAL_DIR, "cases.json")) as f:
        cases = json.load(f)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["turns"] and c.get("category") in run_eval.CATEGORIES, c["id"]


def test_check_case_pass():
    case = {"expect_tools": ["get_order_status"], "expect_args": {"get_order_status": {"order_id": "ORD10002"}},
            "must_match": ["out for delivery"], "forbid_tools": ["get_policy"]}
    checks = run_eval.check_case(case, _transcript("q", "It is out for delivery, Rs 19,999.", [ORDER_CALL]), [ORDER_CALL])
    assert all(c["passed"] for c in checks), checks


def test_check_case_flags_ungrounded_price_and_missing_tool():
    case = {"expect_tools": ["get_order_status"]}
    checks = {c["check"]: c for c in run_eval.check_case(case, _transcript("q", "That costs Rs 12345."), [])}
    assert not checks["calls get_order_status"]["passed"]
    assert not checks["prices come from tool results"]["passed"]
    assert "12345" in checks["prices come from tool results"]["detail"]


def test_derived_prices_are_grounded():
    call = {"name": "get_product_details", "input": {}, "output": {"price": 47990, "mrp": 64900}}
    reply = "Now Rs 47990 (save Rs 16910), or Rs 3999.17/month over 12 months."
    checks = run_eval.check_case({}, _transcript("q", reply, [call]), [call])
    assert all(c["passed"] for c in checks), checks


def test_summarize_and_print(capsys):
    records = [
        {"model": "gemini-2.5-flash", "category": "orders", "passed": True, "error": None, "checks": [{"passed": True}],
         "judge": {"overall": 4.5}, "latency_s": 1.0, "tool_calls": 1},
        {"model": "gemini-2.5-flash", "category": "policies", "passed": False, "error": "boom", "checks": [],
         "judge": None, "latency_s": None, "tool_calls": 0},
    ]
    [row] = run_eval.summarize(records, ["gemini-2.5-flash"])
    assert row["pass_rate"] == 0.5 and row["orders"] == 1 and row["errors"] == 1 and row["judge"] == 4.5
    run_eval.print_table([row], with_judge=True)
    assert "Gemini 2.5 Flash" in capsys.readouterr().out


def test_run_case_end_to_end(fake_client):
    fake_client([completion("Our return window is 7 days.")])
    case = {"id": "p", "category": "policies", "turns": ["return policy?"], "must_match": ["7 days"]}
    record = run_eval.run_case("gemini-2.5-flash", case)
    assert record["error"] is None and record["passed"]


def test_judge_uses_selected_model(fake_client):
    client = fake_client([completion('{"correctness": 5, "helpfulness": 4, "tone": 3, "reason": "ok"}')])
    scores = run_eval.judge("gemini-2.5-flash", {"notes": "n"}, _transcript("q", "a"))
    assert scores["overall"] == 4.0
    assert client.requests[0]["model"] == "gemini-2.5-flash"


def test_judge_on_claude(fake_bedrock):
    client = fake_bedrock([claude_message('{"correctness": 4, "helpfulness": 4, "tone": 4}')])
    scores = run_eval.judge("claude-sonnet-5-5", {"notes": "n"}, _transcript("q", "a"))
    assert scores["overall"] == 4.0
    assert client.requests[0]["model"] == "us.anthropic.claude-sonnet-5-5"
