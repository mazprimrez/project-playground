"""Shorten absolute local paths in notebook outputs (they leak the machine's username into a public repo).

    python scripts/clean_notebook_paths.py notebooks/**/*.ipynb
"""
import json
import re
import sys

PATTERNS = [
    (re.compile(r"/Users/[^/\s\"']+/Documents/(?:GenRec|project-playground)/genrec/"), ""),
    (re.compile(r"/Users/[^/\s\"']+/Documents/(?:GenRec|project-playground)/"), ""),
    (re.compile(r"/private/tmp/[^\s\"']*/scratchpad/"), "<tmp>/"),
    (re.compile(r"/Users/[^/\s\"']+/"), "~/"),
]


def clean(text: str) -> str:
    for pattern, repl in PATTERNS:
        text = pattern.sub(repl, text)
    return text


def main(paths):
    for path in paths:
        nb = json.load(open(path))
        changed = 0
        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                for key in ("text",):
                    if key in out:
                        new = [clean(t) for t in out[key]] if isinstance(out[key], list) else clean(out[key])
                        changed += new != out[key]
                        out[key] = new
                for mime, val in out.get("data", {}).items():
                    if mime.startswith("text"):
                        new = [clean(t) for t in val] if isinstance(val, list) else clean(val)
                        changed += new != val
                        out["data"][mime] = new
                if "traceback" in out:
                    out["traceback"] = [clean(t) for t in out["traceback"]]
        if changed:
            json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
            open(path, "a").write("\n")
        print(f"{path}: {changed} outputs cleaned")


if __name__ == "__main__":
    main(sys.argv[1:])
