import argparse
import json
from pathlib import Path
import sqlite3
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser("Load from json index to sqlite database")

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    with open(args.data_path / "index-raw.json") as f:
        index = json.load(f)
    db_path = args.data_path / "s1" / "index.db"
    db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE results(key, json)")
    to_add = []
    for k, v in index.items():
        to_add.append((k, json.dumps(v["results"])))
    con.executemany("INSERT INTO results VALUES (?, ?)", to_add)
    con.commit()
    con.close()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
