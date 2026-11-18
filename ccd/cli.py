# -*- coding: utf-8 -*-
import argparse
import json
import ccd as m


def main():
    ap = argparse.ArgumentParser(description="ccd")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.path, encoding="utf-8"))
    n = 0
    for q in d["queries"]:
        for t in q["targets"]:
            n += 1
    print(json.dumps({"queries": len(d["queries"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
