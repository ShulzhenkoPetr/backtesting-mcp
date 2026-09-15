from __future__ import annotations

import argparse
from pathlib import Path

from btmcp.eval.report import failures, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    text = render(args.root, args.compare)
    if args.failures:
        text += "\n\n### Failures\n\n" + failures(args.root)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
