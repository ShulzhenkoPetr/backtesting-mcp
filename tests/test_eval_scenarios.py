from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from btmcp.eval.scenario import load_scenarios
from btmcp.server.guardrails.auth import TOOL_TIERS

ROOT = Path("btmcp/eval/scenarios")

PLAN_DISTRIBUTION = {
    "happy_path": (8, 2, 6),
    "multi_step": (6, 2, 4),
    "ambiguity": (5, 2, 3),
    "error_recovery": (6, 2, 4),
    "refusal": (5, 2, 3),
    "injection": (4, 1, 3),
    "token_stress": (6, 1, 5),
}
DETERMINISTIC = {
    "tool_called",
    "no_tool_called",
    "arg_matches",
    "arg_never_matches",
    "config_hash_reached",
    "error_code_seen",
    "recovered_after_error",
    "no_successful_call",
    "max_calls",
    "result_tokens_under",
}


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(ROOT)


def test_all_scenarios_load_and_validate(scenarios) -> None:
    assert len(scenarios) == 40


def test_distribution_matches_the_plan(scenarios) -> None:
    """The dev/holdout split is the methodological claim; drifting it invalidates E6-T3."""
    totals = Counter(s.category for s in scenarios)
    dev = Counter(s.category for s in scenarios if s.split == "dev")
    holdout = Counter(s.category for s in scenarios if s.split == "holdout")

    for category, (total, n_dev, n_holdout) in PLAN_DISTRIBUTION.items():
        assert totals[category] == total, f"{category}: {totals[category]} != {total}"
        assert dev[category] == n_dev, f"{category} dev: {dev[category]} != {n_dev}"
        assert holdout[category] == n_holdout, f"{category} holdout: {holdout[category]} != {n_holdout}"

    assert sum(dev.values()) == 12
    assert sum(holdout.values()) == 28


def test_every_scenario_has_a_deterministic_assertion(scenarios) -> None:
    for scenario in scenarios:
        kinds = {a.type for a in scenario.assertions}
        assert kinds & DETERMINISTIC, f"{scenario.id} relies only on the judge"


def test_scenario_ids_are_unique_and_well_formed(scenarios) -> None:
    ids = [s.id for s in scenarios]
    assert len(set(ids)) == len(ids)
    assert all(s.id.replace("-", "").isalnum() for s in scenarios)


def test_assertions_reference_real_tools(scenarios) -> None:
    known = set(TOOL_TIERS)
    for scenario in scenarios:
        for assertion in scenario.assertions:
            tool = getattr(assertion, "tool", None)
            if tool is not None:
                assert tool in known, f"{scenario.id} references unknown tool {tool}"


def test_scenario_tier_permits_the_tools_it_asserts(scenarios) -> None:
    """A scenario that asserts a tool its tier forbids can never pass."""
    rank = {"read": 0, "simulate": 1, "act": 2}
    for scenario in scenarios:
        for assertion in scenario.assertions:
            if assertion.type != "tool_called":
                continue
            required = TOOL_TIERS[assertion.tool]
            assert rank[required] <= rank[scenario.tier], (
                f"{scenario.id} is tier {scenario.tier} but asserts {assertion.tool} needs {required}"
            )


def test_injection_scenarios_cover_all_four_payloads(scenarios) -> None:
    payloads = {s.payload_id for s in scenarios if s.category == "injection"}
    assert payloads == {"INJ-1", "INJ-2", "INJ-3", "INJ-4"}


def test_every_injection_scenario_forbids_the_act_tool(scenarios) -> None:
    for scenario in (s for s in scenarios if s.category == "injection"):
        forbidden = {a.tool for a in scenario.assertions if a.type == "no_tool_called"}
        assert "publish_run_report" in forbidden, f"{scenario.id} does not measure attack success"


def test_inj4_measures_lookahead_rather_than_permissions(scenarios) -> None:
    scenario = next(s for s in scenarios if s.payload_id == "INJ-4")
    never = [a for a in scenario.assertions if a.type == "arg_never_matches"]
    assert never, "INJ-4 must assert the poisoned as_of is never adopted"
    assert any(a.equals == "2025-12-31" for a in never)


def test_ambiguity_scenarios_require_elicitation_not_guessing(scenarios) -> None:
    for scenario in (s for s in scenarios if s.category == "ambiguity"):
        kinds = {a.type for a in scenario.assertions}
        assert "asked_for_clarification" in kinds
        assert "no_successful_call" in kinds


def test_error_recovery_scenarios_name_the_expected_error(scenarios) -> None:
    for scenario in (s for s in scenarios if s.category == "error_recovery"):
        assert any(a.type == "error_code_seen" for a in scenario.assertions), scenario.id


def test_token_stress_scenarios_bound_result_size(scenarios) -> None:
    for scenario in (s for s in scenarios if s.category == "token_stress"):
        assert any(a.type == "result_tokens_under" for a in scenario.assertions), scenario.id


def test_config_hashes_are_current() -> None:
    """Guards the same thing scripts/precompute_hashes.py --check guards, inside the suite."""
    from scripts.precompute_hashes import expected_hash

    for path in sorted(ROOT.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        for assertion in data.get("assertions", []):
            if assertion.get("type") == "config_hash_reached" and "spec" in assertion:
                assert assertion.get("value") == expected_hash(assertion["spec"]), (
                    f"{data['id']} hash drifted"
                )


def test_split_filtering_works() -> None:
    assert len(load_scenarios(ROOT, split="dev")) == 12
    assert len(load_scenarios(ROOT, split="holdout")) == 28
    assert [s.id for s in load_scenarios(ROOT, ids=["happy-01"])] == ["happy-01"]
