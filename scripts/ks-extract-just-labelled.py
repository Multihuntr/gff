import argparse
import tarfile
import sys
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser("From kurosiwo archive, extract only labelled subset")

    parser.add_argument("tar_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    with tarfile.open(args.tar_path) as f:
        print("Searching for targets...")
        w_targets = {}
        members = []
        for i, m in enumerate(f):
            p = Path(m.path).parent
            members.append(m)
            if i % 1000 == 0:
                print(f"{i:7d} files - {len(w_targets):4d} target folders found", end="\r")
            if "MK0_MLU_" in m.path:
                # MLU is the labels file. Unlabelled regions do not have it.
                w_targets[p] = True
        print()

        def filter_targs(mem, p):
            if Path(mem.path).parent in w_targets:
                return mem
            else:
                return None

        print("Extracting just the targets...")
        for i, m in enumerate(members):
            if i % 1000 == 0:
                print(f"{i:7d} files extracted", end="\r")
            if filter_targs(m, None):
                f.extract(m, args.tar_path.parent, filter=lambda m, p: m)
        print()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
