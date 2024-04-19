import argparse
from pathlib import Path
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser("Filter by basin x floods manually filtered")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("incl_list_path", type=Path)
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args(argv)


def main(args):
    incl = {}
    with open(args.incl_list_path) as f:
        for line in f:
            incl[line.strip()] = True

    for path in (args.data_path / "floodmaps").glob("*/????-??????????-*"):
        key = "-".join(path.name.split("-")[:2])
        if key not in incl:
            if args.dry_run:
                print(f"Will remove {path}")
            else:
                path.unlink()
    for path in (args.data_path / "s1").glob("????-??????????-*"):
        key = "-".join(path.name.split("-")[:2])
        if key not in incl:
            if args.dry_run:
                print(f"Will remove {path}")
            else:
                path.unlink()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
