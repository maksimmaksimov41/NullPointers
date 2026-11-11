"""Generate a reproducible practice set or score a completed local run."""
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys


def generate(directory, seed, count):
    directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    expected = {}
    for index in range(count):
        kind = ("math", "data_analysis", "search", "coding", "missing")[index % 5]
        root = directory / f"{kind}_{index:04d}"
        root.mkdir()
        workspace = root / "workspace"
        workspace.mkdir()
        reward = rng.randint(1, 10)
        if kind == "math":
            boxes, rate, days, deliveries, size = (rng.randint(a, b) for a, b in (
                (500, 5000), (2, 20), (2, 10), (2, 6), (5, 50)
            ))
            instruction = (
                f"A warehouse starts with {boxes} boxes. It ships {rate} boxes per day for {days} days. "
                f"It receives {deliveries} deliveries of {size} boxes each. How many boxes does it have at the end?"
            )
            answer = str(boxes - rate * days + deliveries * size)
        elif kind == "data_analysis":
            first = rng.randint(1, 10)
            rows = [(rng.choice(("widget", "sprocket")), rng.randint(1, 12), rng.randint(1, 50))
                    for _ in range(rng.randint(20, 300))]
            with (workspace / "sales.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("product", "month", "units"))
                writer.writerows(rows)
            instruction = f"How many units of product 'widget' were sold in months {first} through {first + 1} inclusive?"
            answer = str(sum(units for product, month, units in rows
                             if product == "widget" and first <= month <= first + 1))
        elif kind == "search":
            year = rng.randint(1950, 2000)
            kb = workspace / "kb"
            kb.mkdir()
            (kb / "person_ren_sato.md").write_text(f"# Ren Sato\n\nBorn: {year}\n", encoding="utf-8")
            (kb / "person_ren_ford.md").write_text(f"# Ren Ford\n\nBorn: {year - 1}\n", encoding="utf-8")
            instruction = "The directory kb/ contains Markdown records. In which year was Ren Sato born?"
            answer = str(year)
        elif kind == "coding":
            values = [rng.randint(-500, 500) for _ in range(6)]
            (workspace / "scores.py").write_text(
                "def positive_total(values):\n    return sum(value for value in values if value < 0)\n",
                encoding="utf-8",
            )
            (workspace / "test_scores.py").write_text(
                "import unittest\nfrom scores import positive_total\n\n"
                "class TestScores(unittest.TestCase):\n    def test_positive_total(self):\n"
                f"        self.assertEqual(positive_total({values + [1, -1]}), "
                f"{1 + sum(value for value in values if value > 0)})\n"
                "        self.assertEqual(positive_total([-5, -1, 0]), 0)\n",
                encoding="utf-8",
            )
            instruction = "Fix positive_total in scores.py. Run the unittest suite and write DONE after it passes."
            answer = "DONE"
        else:
            kind = "data_analysis"
            instruction = "The file sales.csv contains sales records. What is the total number of units sold?"
            answer = None

        (root / "task.json").write_text(json.dumps({
            "task_id": root.name, "task_type": kind, "instruction": instruction, "reward": reward,
        }), encoding="utf-8")
        expected[root.name] = {"answer": answer, "reward": reward}

    directory.with_name(directory.name + ".expected.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )


def score(directory):
    expected = json.loads(directory.with_name(directory.name + ".expected.json").read_text(encoding="utf-8"))
    points = correct = wrong = skipped = impossible_skipped = 0
    for name, spec in expected.items():
        root = directory / name
        path = root / "answer.txt"
        expected_answer = spec["answer"]
        if not path.is_file():
            skipped += 1
            impossible_skipped += expected_answer is None
            continue
        valid = path.read_text(encoding="utf-8").strip() == expected_answer
        if valid and expected_answer == "DONE":
            tests = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "discover"],
                cwd=root / "workspace", capture_output=True, timeout=8,
            )
            valid = tests.returncode == 0
        if valid:
            correct += 1
            points += spec["reward"]
        else:
            wrong += 1
    possible = sum(spec["reward"] for spec in expected.values() if spec["answer"] is not None)
    print(json.dumps({
        "correct": correct, "wrong": wrong, "skipped": skipped,
        "impossible_skipped": impossible_skipped, "points": points, "possible_points": possible,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "score"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.action == "generate":
        generate(args.directory, args.seed, args.count)
    else:
        score(args.directory)

