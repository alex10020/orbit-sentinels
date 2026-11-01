"""Run every named query in sql/03_validation.sql and print (or save) the results.

    python pipeline/validate.py --db db/orbital_sentinel.db [--out results/validation.txt]
"""
import argparse
import sqlite3
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def queries(text: str):
    name, buf = None, []
    for line in text.splitlines():
        if line.startswith("-- name:"):
            if name and "".join(buf).strip():
                yield name, "\n".join(buf)
            name, buf = line[len("-- name:"):].strip(), []
        elif name:
            buf.append(line)
    if name and "".join(buf).strip():
        yield name, "\n".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    lines = []
    for name, sql in queries((HERE / "sql" / "03_validation.sql").read_text()):
        t0 = time.perf_counter()
        cur = con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        lines.append(f"\n== {name}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        lines.append(" | ".join(cols))
        lines += [" | ".join(str(v) for v in r) for r in rows]
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
