from __future__ import annotations

import argparse
from pathlib import Path

from btmcp.data.manifest import (
    DATASET_ID,
    GLOBAL_SEED,
    UNIVERSE,
    load_manifest,
    save_manifest,
)
from btmcp.data.synthetic import (
    GENERATOR_VERSION,
    Manifest,
    SymbolRecord,
    generate_symbol,
    sha256_file,
    write_symbol,
)

DEFAULT_ROOT = Path("datasets/synthetic")


def build(root: Path) -> Manifest:
    root.mkdir(parents=True, exist_ok=True)
    records: list[SymbolRecord] = []
    for params in UNIVERSE:
        frame = generate_symbol(params, GLOBAL_SEED)
        path = write_symbol(frame, root)
        records.append(
            SymbolRecord(
                params=params,
                n_rows=frame.height,
                first=frame["timestamp"][0],
                last=frame["timestamp"][-1],
                sha256=sha256_file(path),
            )
        )
    return Manifest(
        generator_version=GENERATOR_VERSION,
        global_seed=GLOBAL_SEED,
        dataset_id=DATASET_ID,
        symbols=records,
    )


def check(root: Path) -> int:
    recorded = load_manifest(root)
    if recorded.generator_version != GENERATOR_VERSION:
        print(f"generator version drift: manifest {recorded.generator_version} != code {GENERATOR_VERSION}")
        return 1
    failures = 0
    for record in recorded.symbols:
        path = root / f"{record.params.symbol}.parquet"
        if not path.exists():
            print(f"MISSING {path}")
            failures += 1
            continue
        actual = sha256_file(path)
        if actual != record.sha256:
            print(f"HASH MISMATCH {record.params.symbol}: {actual[:16]} != {record.sha256[:16]}")
            failures += 1
    print(f"{len(recorded.symbols) - failures}/{len(recorded.symbols)} symbols verified")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify the synthetic dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true", help="verify hashes instead of regenerating")
    args = parser.parse_args()

    if args.check:
        return check(args.root)

    manifest = build(args.root)
    save_manifest(manifest, args.root)
    total = sum(r.n_rows for r in manifest.symbols)
    print(f"wrote {len(manifest.symbols)} symbols, {total} rows -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
