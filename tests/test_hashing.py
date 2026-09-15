from __future__ import annotations

import json
import subprocess
import sys

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from btmcp.core.hashing import canonical_json, config_hash

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=20),
)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)
configs = st.fixed_dictionaries(
    {
        "spec": st.dictionaries(st.text(max_size=8), json_values, max_size=5),
        "dataset_id": st.text(max_size=12),
        "costs": st.dictionaries(st.text(max_size=8), json_values, max_size=5),
    }
)


def _shuffled(value: object, rng_seed: int) -> object:
    """Rebuilds nested dicts with permuted insertion order."""
    import random

    rng = random.Random(rng_seed)
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {k: _shuffled(v, rng.randint(0, 10**6)) for k, v in items}
    if isinstance(value, list):
        return [_shuffled(v, rng.randint(0, 10**6)) for v in value]
    return value


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(cfg=configs, seed=st.integers(min_value=0, max_value=10**6))
def test_hash_is_insertion_order_invariant(cfg: dict, seed: int) -> None:
    permuted = _shuffled(cfg, seed)
    assert isinstance(permuted, dict)
    assert config_hash(**cfg) == config_hash(**permuted)


@settings(max_examples=100)
@given(cfg=configs)
def test_hash_is_deterministic_within_process(cfg: dict) -> None:
    assert config_hash(**cfg) == config_hash(**cfg)


@settings(max_examples=50)
@given(cfg=configs)
def test_canonical_json_is_valid_json(cfg: dict) -> None:
    json.loads(canonical_json(cfg))


def test_hash_is_stable_across_processes() -> None:
    """PYTHONHASHSEED varies per process; a dict-iteration-dependent hash would drift."""
    cases = [
        {
            "spec": {"primitive": "sma_cross", "fast": 20, "slow": 50},
            "dataset_id": "syn-v1",
            "costs": {"fee_bps": 5.0},
        },
        {"spec": {"z": [1, 2.5, {"k": None}], "a": True}, "dataset_id": "syn-v1", "costs": {"fee_bps": 0.0}},
    ]
    src = (
        "import json,sys;from btmcp.core.hashing import config_hash;"
        "print(json.dumps([config_hash(**c) for c in json.loads(sys.argv[1])]))"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", src, json.dumps(cases)],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        outs.add(proc.stdout.strip())
    assert len(outs) == 1, f"hash drifted across processes: {outs}"
    assert [config_hash(**c) for c in cases] == json.loads(outs.pop())


def test_engine_version_participates() -> None:
    base = {"spec": {"a": 1}, "dataset_id": "d", "costs": {"fee_bps": 1.0}}
    assert config_hash(**base, engine_version="1.0.0") != config_hash(**base, engine_version="1.0.1")
