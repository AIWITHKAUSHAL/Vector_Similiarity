"""Refresh the standalone visualizer's embedded report and source snapshots."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "architecture_visualizer.html"
START = '<script id="project-data" type="application/json">'
END = "</script><!-- project-data:end -->"


def main():
    data = {
        "report": json.loads(
            (ROOT / "experiment_results.json").read_text(encoding="utf-8")
        ),
        "sources": {
            name: (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "app.py",
                "retrieval.py",
                "run_experiments.py",
                "tests/test_retrieval.py",
            )
        },
    }
    before, rest = TARGET.read_text(encoding="utf-8").split(START, 1)
    _, after = rest.split(END, 1)
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    TARGET.write_text(before + START + payload + END + after, encoding="utf-8")
    print(f"Updated {TARGET}")


if __name__ == "__main__":
    main()
