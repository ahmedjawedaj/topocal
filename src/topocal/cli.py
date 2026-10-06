"""Command line interface."""

from __future__ import annotations

import argparse
import json

from topocal.demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="topocal", description="TopoCal research CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the synthetic routing vertical slice")
    inspect = sub.add_parser(
        "inspect-shard",
        help="print the observed HDF5 layout of a TopoBox-3D shard (needs topocal[data])",
    )
    inspect.add_argument("path", help="path to a shard_NNNN.h5 file")
    inspect.add_argument(
        "--kind",
        choices=("geometry", "hodge_heat"),
        default=None,
        help="compare the layout with the documented one",
    )
    inspect.add_argument("--sha256", action="store_true", help="also hash the file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(), indent=2))
    elif args.command == "inspect-shard":
        from topocal.datasets.topobox_schema import inspect_shard

        schema = inspect_shard(args.path, kind=args.kind, hash_file=args.sha256)
        print(json.dumps(schema.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
