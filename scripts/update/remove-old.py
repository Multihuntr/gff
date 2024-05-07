import argparse
from pathlib import Path
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser("Filter by basin x floods manually filtered")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("--incl_list_path", default=None, type=Path)
    parser.add_argument("--excl_list_path", default=None, type=Path)
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args(argv)


def main(args):
    use_incl = args.incl_list_path is not None
    use_excl = args.excl_list_path is not None
    assert (use_incl or use_excl) and (
        use_incl != use_excl
    ), "Must provide either incl list or excl."
    if use_incl:
        incl = {}
        with open(args.incl_list_path) as f:
            for line in f:
                incl[line.strip()] = True
    else:
        excl = {}
        with open(args.excl_list_path) as f:
            for line in f:
                excl[line.strip()] = True

    for path in (args.data_path / "floodmaps").glob("*/????-??????????-*"):
        key = "-".join(path.name.split("-")[:2])
        if use_incl:
            should_remove = key not in incl
        else:
            should_remove = key in excl
        if should_remove:
            if args.dry_run:
                print(f"Will remove {path}")
            else:
                pass
                path.unlink()
    for path in (args.data_path / "s1").glob("????-??????????-*"):
        key = "-".join(path.name.split("-")[:2])
        if use_incl:
            should_remove = key not in incl
        else:
            should_remove = key in excl
        if should_remove:
            if args.dry_run:
                print(f"Will remove {path}")
            else:
                pass
                path.unlink()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
