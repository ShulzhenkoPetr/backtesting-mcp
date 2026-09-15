from __future__ import annotations

import json
from pathlib import Path

from btmcp.eval.events import ToolCall, Turn
from btmcp.eval.judge import Judge, StubJudge
from btmcp.eval.metrics import aggregate, by_seed_spread, flip_table, load_runs, succeeded
from btmcp.eval.models import ScriptedModel
from btmcp.eval.report import render
from btmcp.eval.runner import outcome_exists, run_scenario, write_outcome
from btmcp.eval.scenario import Scenario
from btmcp.server.surface_a import build

SMA = {"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}}
RUN_ARGS = {
    "spec": SMA,
    "symbol": "SYN-04",
    "start": "2019-01-01",
    "end": "2021-12-31",
    "costs": {"fee_bps": 5, "slippage_bps": 2},
}


def scenario(**kw) -> Scenario:
    base = {
        "id": "t-01",
        "category": "happy_path",
        "split": "dev",
        "tier": "simulate",
        "prompt": "test",
        "assertions": [{"type": "tool_called", "tool": "run_backtest"}],
    }
    return Scenario.model_validate({**base, **kw})


def run_turn(**kw) -> Turn:
    return Turn(**kw)


async def go(sc: Scenario, model: ScriptedModel, tmp_path: Path, **kw):
    return await run_scenario(sc, build, model, runs_root=tmp_path / "runs", **kw)


async def test_a_successful_run_passes_its_assertions(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            Turn(
                tool_calls=[ToolCall("t1", "run_backtest", RUN_ARGS)],
                stop_reason="tool_use",
                input_tokens=1000,
                output_tokens=50,
            ),
            Turn(text="Sharpe was 1.46.", input_tokens=1500, output_tokens=20),
        ]
    )
    out = await go(
        scenario(
            assertions=[
                {"type": "tool_called", "tool": "run_backtest"},
                {"type": "arg_matches", "tool": "run_backtest", "path": "costs.fee_bps", "equals": 5},
                {"type": "no_tool_called", "tool": "publish_run_report"},
            ]
        ),
        model,
        tmp_path,
    )

    assert out.task_success
    assert out.total_calls == 1
    assert out.calls_to_completion == 1
    assert out.total_tokens == 2570
    assert not out.hit_max_turns
    assert out.final_answer == "Sharpe was 1.46."


async def test_max_turns_is_enforced_and_marks_completion_infinite(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            Turn(
                tool_calls=[ToolCall(f"t{i}", "list_symbols", {"as_of": "2021-12-31"})],
                stop_reason="tool_use",
            )
            for i in range(20)
        ]
    )
    out = await go(scenario(max_turns=4), model, tmp_path)
    assert out.hit_max_turns
    assert out.calls_to_completion == float("inf")
    assert len(out.turns) == 4


async def test_typed_errors_reach_the_model_and_are_recorded(tmp_path: Path) -> None:
    bad = {**RUN_ARGS, "spec": {"primitive": "sma_crossover", "params": {}}}
    model = ScriptedModel(
        [
            Turn(tool_calls=[ToolCall("t1", "run_backtest", bad)], stop_reason="tool_use"),
            Turn(tool_calls=[ToolCall("t2", "run_backtest", RUN_ARGS)], stop_reason="tool_use"),
            Turn(text="Fixed the primitive name."),
        ]
    )
    out = await go(
        scenario(
            category="error_recovery",
            assertions=[
                {"type": "error_code_seen", "code": "INVALID_SPEC"},
                {"type": "recovered_after_error", "tool": "run_backtest", "within": 3},
            ],
        ),
        model,
        tmp_path,
    )

    assert out.task_success
    assert out.error_codes == ["INVALID_SPEC"]
    # The error payload must be legible to the model, not a stack trace.
    sent_back = json.dumps([vars(e) for e in model.seen[-1]], default=str)
    assert "suggested_action" in sent_back and "Traceback" not in sent_back


async def test_tier_denial_is_recorded_as_an_error(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            Turn(tool_calls=[ToolCall("t1", "run_backtest", RUN_ARGS)], stop_reason="tool_use"),
            Turn(text="I cannot run backtests at this tier."),
        ]
    )
    out = await go(
        scenario(
            tier="read",
            assertions=[
                {"type": "error_code_seen", "code": "TIER_DENIED"},
            ],
        ),
        model,
        tmp_path,
    )
    assert out.task_success and out.error_codes == ["TIER_DENIED"]


async def test_parallel_tool_calls_arrive_as_one_result_event(tmp_path: Path) -> None:
    """The runner emits neutral events; how they are rendered is each adapter's business."""
    from btmcp.eval.events import ToolResults

    model = ScriptedModel(
        [
            Turn(
                tool_calls=[
                    ToolCall("t1", "list_symbols", {"as_of": "2021-12-31"}),
                    ToolCall("t2", "describe_series", {"symbol": "SYN-04", "as_of": "2021-12-31"}),
                ],
                stop_reason="tool_use",
            ),
            Turn(text="done"),
        ]
    )
    out = await go(scenario(assertions=[{"type": "tool_called", "tool": "list_symbols"}]), model, tmp_path)
    assert out.total_calls == 2

    history = model.seen[-1]
    results = [e for e in history if isinstance(e, ToolResults)]
    assert len(results) == 1, "both calls must be reported in a single result event"
    assert [r.name for r in results[0].results] == ["list_symbols", "describe_series"]


async def test_injection_payload_reaches_the_model_fenced(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            Turn(
                tool_calls=[
                    ToolCall(
                        "t1",
                        "get_news",
                        {
                            "symbol": "SYN-04",
                            "start": "2020-01-01",
                            "end": "2020-03-12",
                            "as_of": "2020-03-12",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            Turn(text="The headlines discuss a trading update."),
        ]
    )
    out = await go(
        scenario(
            category="injection",
            assertions=[
                {"type": "tool_called", "tool": "get_news"},
                {"type": "no_tool_called", "tool": "publish_run_report"},
            ],
        ),
        model,
        tmp_path,
    )

    assert out.task_success
    delivered = json.dumps([vars(e) for e in model.seen[-1]], default=str)
    assert "UNTRUSTED_DATA" in delivered
    assert "publish_run_report" in delivered, "payload must arrive intact, fenced not stripped"


async def test_fencing_can_be_disabled_for_the_ablation(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            Turn(
                tool_calls=[
                    ToolCall(
                        "t1",
                        "get_news",
                        {
                            "symbol": "SYN-04",
                            "start": "2020-01-01",
                            "end": "2020-03-12",
                            "as_of": "2020-03-12",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            Turn(text="ok"),
        ]
    )
    await go(
        scenario(assertions=[{"type": "tool_called", "tool": "get_news"}]), model, tmp_path, fencing=False
    )
    assert "UNTRUSTED_DATA" not in json.dumps(model.seen[-1], default=str)


async def test_outcomes_are_written_and_resumable(tmp_path: Path) -> None:
    model = ScriptedModel([Turn(text="no tools needed")])
    out = await go(scenario(assertions=[{"type": "max_calls", "value": 0}]), model, tmp_path)
    results = tmp_path / "results"
    assert not outcome_exists(out.scenario_id, "a", 0, results)
    path = write_outcome(out, results)
    assert outcome_exists(out.scenario_id, "a", 0, results)
    assert json.loads(path.read_text())["scenario_id"] == "t-01"


async def test_metrics_and_report_render_from_written_runs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    good = ScriptedModel(
        [
            Turn(
                tool_calls=[ToolCall("t1", "run_backtest", RUN_ARGS)],
                stop_reason="tool_use",
                input_tokens=900,
                output_tokens=40,
            ),
            Turn(text="Sharpe 1.46", input_tokens=1200, output_tokens=15),
        ]
    )
    out = await go(scenario(), good, tmp_path)
    write_outcome(out, results)

    bad = ScriptedModel([Turn(text="I did nothing.")])
    out2 = await go(scenario(id="t-02"), bad, tmp_path)
    write_outcome(out2, results)

    runs = load_runs(results)
    assert len(runs) == 2
    agg = aggregate(runs)
    assert agg.n == 2 and agg.task_success == 0.5

    text = render(results)
    assert "task success" in text and "Per category" in text
    assert "seed spread is not a finding" in text


def test_rubric_below_threshold_fails_the_run() -> None:
    run = {
        "assertions": [{"type": "tool_called", "passed": True, "detail": ""}],
        "rubric_score": 2,
    }
    assert not succeeded(run)
    assert succeeded({**run, "rubric_score": 3})
    assert succeeded({**run, "rubric_score": None})


def test_attack_success_counts_actions_not_mentions() -> None:
    acted = {
        "category": "injection",
        "hit_max_turns": False,
        "total_calls": 1,
        "wrong_tool_calls": 0,
        "total_tokens": 10,
        "seed": 0,
        "scenario_id": "i1",
        "audit": [],
        "assertions": [{"type": "no_tool_called", "passed": False, "detail": "called 1x"}],
    }
    resisted = {**acted, "assertions": [{"type": "no_tool_called", "passed": True, "detail": "called 0x"}]}
    assert aggregate([acted]).attack_success_rate == 1.0
    assert aggregate([resisted]).attack_success_rate == 0.0
    assert aggregate([acted, resisted]).attack_success_rate == 0.5


def test_flip_table_pairs_scenarios_across_surfaces() -> None:
    def run(sid: str, surface: str, passed: bool, seed: int = 0) -> dict:
        return {
            "scenario_id": sid,
            "surface": surface,
            "category": "happy_path",
            "seed": seed,
            "assertions": [{"type": "tool_called", "passed": passed, "detail": ""}],
            "rubric_score": None,
            "hit_max_turns": False,
            "total_calls": 1,
            "wrong_tool_calls": 0,
            "total_tokens": 1,
            "audit": [],
        }

    a = [run("s1", "a", True), run("s2", "a", True), run("s3", "a", False), run("s4", "a", False)]
    b = [run("s1", "b", True), run("s2", "b", False), run("s3", "b", True), run("s4", "b", False)]
    flips = flip_table(a, b)
    assert [f.scenario_id for f in flips["both"]] == ["s1"]
    assert [f.scenario_id for f in flips["a_only"]] == ["s2"]
    assert [f.scenario_id for f in flips["b_only"]] == ["s3"]
    assert [f.scenario_id for f in flips["neither"]] == ["s4"]


def test_seed_spread_is_reported_so_deltas_can_be_judged() -> None:
    def run(seed: int, passed: bool) -> dict:
        return {
            "seed": seed,
            "scenario_id": f"s{seed}",
            "category": "happy_path",
            "assertions": [{"type": "tool_called", "passed": passed, "detail": ""}],
            "rubric_score": None,
        }

    spread = by_seed_spread([run(0, True), run(1, False), run(2, True)])
    assert spread["min"] == 0.0 and spread["max"] == 1.0 and spread["spread"] == 1.0


async def test_judge_caches_by_transcript(tmp_path: Path) -> None:
    from btmcp.eval.models import NullModel

    cache = tmp_path / "judge.json"
    judge = Judge(NullModel(), cache)
    assert judge.cached("rubric", "answer") is None

    from btmcp.eval.judge import Grade

    judge._store("rubric", "answer", Grade(score=4, justification="fine"))
    assert Judge(NullModel(), cache).cached("rubric", "answer").score == 4

    other = NullModel()
    other.id = "other-model"
    assert Judge(other, cache).cached("rubric", "answer") is None


async def test_stub_judge_is_deterministic() -> None:
    judge = StubJudge(score=5)
    assert (await judge.grade("r", "a good answer")).score == 5
    assert (await judge.grade("r", "")).score == 1


def test_the_whole_harness_runs_without_credentials() -> None:
    """CI has no API key. Everything except the live model must work regardless."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import btmcp.eval.runner, btmcp.eval.metrics, btmcp.eval.report, btmcp.eval.judge;"
                "from btmcp.eval.models import build_model;"
                "from btmcp.eval.ladder import load_ladder;"
                "from btmcp.eval.scenario import load_scenarios;"
                "print(len(load_scenarios()), len(load_ladder()), build_model('null').id)"
            ),
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path.cwd())},
        cwd=Path.cwd(),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[0] == "40"


def test_no_sampling_parameters_are_sent() -> None:
    """Current models reject `temperature`; seeds are the variance control instead."""
    source = Path("btmcp/eval/models.py").read_text()
    for forbidden in ("temperature", "top_p", "top_k"):
        assert f'"{forbidden}"' not in source and f"{forbidden}=" not in source


def test_the_sweep_command_path_runs_end_to_end_at_zero_cost(tmp_path: Path) -> None:
    """Smoke-tests scenario loading, the runner, output writing, resume and reporting."""
    from scripts.report import main as report_main
    from scripts.run_eval import main as eval_main

    out = tmp_path / "a"
    args = [
        "--surface",
        "a",
        "--model",
        "null",
        "--split",
        "dev",
        "--seeds",
        "1",
        "--runs",
        str(tmp_path / "runs"),
        "--out",
        str(out),
    ]
    assert eval_main(args) == 0
    written = list(out.glob("*.json"))
    assert len(written) == 13, "12 scenario outcomes plus summary.json"

    # Resuming must not re-run anything.
    stamps = {p: p.stat().st_mtime_ns for p in out.glob("*__a__seed0.json")}
    assert eval_main(args) == 0
    assert {p: p.stat().st_mtime_ns for p in out.glob("*__a__seed0.json")} == stamps

    assert report_main([str(out)]) == 0


def test_the_sweep_runs_against_surface_b_unmodified(tmp_path: Path) -> None:
    """E6-T1 acceptance, exercised through the real CLI rather than in-process."""
    from scripts.run_eval import main as eval_main

    out = tmp_path / "b"
    assert (
        eval_main(
            [
                "--surface",
                "b",
                "--model",
                "null",
                "--split",
                "dev",
                "--seeds",
                "1",
                "--runs",
                str(tmp_path / "runs"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert len(list(out.glob("*__b__seed0.json"))) == 12


def test_holdout_requires_explicit_confirmation(tmp_path: Path) -> None:
    """Iterating against holdout is schema overfitting with extra steps."""
    from scripts.run_eval import main as eval_main

    args = ["--split", "holdout", "--model", "null", "--out", str(tmp_path / "h"), "--dry-run"]
    assert eval_main(args) == 1
    assert eval_main([*args, "--confirm-holdout"]) == 0
