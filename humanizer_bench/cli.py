"""Command-line interface for humanizer-bench.

Installed as the ``humanizer-bench`` console script. Repeat ``--detector`` /
``--attack`` to evaluate a full detector × attack matrix; a single pair prints
the detailed before/after table instead.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .registry import ATTACKS, DATASETS, DETECTORS, get_attack, get_dataset, get_detector
from .runner import format_matrix, format_result, run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanizer-bench",
        description="Benchmark the robustness of AI-text detectors against humanizer attacks.",
    )
    parser.add_argument(
        "-d",
        "--detector",
        action="append",
        metavar="NAME",
        help="detector to evaluate; repeat for a matrix (default: heuristic)",
    )
    parser.add_argument(
        "-a",
        "--attack",
        action="append",
        metavar="NAME",
        help="attack to apply to AI texts; repeat for a matrix (default: sentence_merge)",
    )
    parser.add_argument(
        "--dataset",
        default="toy",
        metavar="NAME",
        help="dataset to evaluate on (default: toy)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="decision threshold on the AI score (default: 0.5)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list registered components and exit"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _list_components() -> str:
    lines = ["detectors:"]
    lines += [f"  {name}" for name in sorted(DETECTORS)]
    lines.append("attacks:")
    lines += [f"  {name}" for name in sorted(ATTACKS)]
    lines.append("datasets:")
    lines += [f"  {name}" for name in sorted(DATASETS)]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(_list_components())
        return 0

    detector_names = args.detector or ["heuristic"]
    attack_names = args.attack or ["sentence_merge"]

    try:
        detectors = [get_detector(name) for name in detector_names]
        attacks = [get_attack(name) for name in attack_names]
        dataset = get_dataset(args.dataset)
        results = run_matrix(detectors, attacks, dataset=dataset, threshold=args.threshold)
    except ValueError as err:  # unknown component name
        print(f"error: {err}", file=sys.stderr)
        return 2
    except NotImplementedError as err:  # a Phase 2 stub was selected
        print(f"error: {err}", file=sys.stderr)
        return 2

    if len(results) == 1:
        print(format_result(results[0]))
    else:
        print(format_matrix(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
