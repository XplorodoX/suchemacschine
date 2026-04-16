"""
Evaluation script for the HS Aalen search engine.
Runs test queries against the live search API and scores result quality.

Usage:
    python eval.py                          # Run against localhost:8000
    python eval.py --url http://host:8080  # Custom API URL
    python eval.py --top-k 5               # Check top-5 instead of top-3
"""

import json
import argparse
import unicodedata
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


TEST_FILE = Path(__file__).parent / "test_queries.json"


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def check_result(result: dict, test: dict) -> dict:
    """Returns a dict with pass/fail per criterion."""
    checks = {}

    text = normalize(result.get("text", ""))
    url = normalize(result.get("url", "") or "")
    result_type = result.get("type", "")

    if "expected_url_contains" in test:
        checks["url_match"] = any(kw in url for kw in test["expected_url_contains"])

    if "expected_keywords" in test:
        checks["keyword_match"] = any(kw in text or kw in url for kw in test["expected_keywords"])

    if "expected_type" in test:
        checks["type_match"] = result_type == test["expected_type"]

    return checks


def score_query(api_url: str, test: dict, top_k: int = 3) -> dict:
    """Run a single test query and score the top-k results."""
    query = test["query"]
    try:
        resp = requests.get(
            f"{api_url}/api/search",
            params={"q": query, "per_page": top_k, "provider": "none", "include_rerank": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"query": query, "error": str(e), "passed": False}

    results = data.get("results", [])
    if not results:
        return {"query": query, "passed": False, "reason": "no results", "top_results": []}

    # A test passes if ANY of the top-k results satisfies ALL its criteria
    any_pass = False
    per_result = []
    for r in results[:top_k]:
        checks = check_result(r, test)
        passed = all(checks.values()) if checks else False
        per_result.append({"url": r.get("url"), "score": r.get("score"), "checks": checks, "passed": passed})
        if passed:
            any_pass = True

    return {
        "query": query,
        "category": test.get("category", ""),
        "passed": any_pass,
        "top_results": per_result,
    }


def run_eval(api_url: str, top_k: int):
    tests = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    print(f"Running {len(tests)} test queries against {api_url} (top-{top_k})\n")
    print(f"{'Query':<45} {'Category':<20} {'Result'}")
    print("-" * 80)

    passed = 0
    by_category: dict[str, list[bool]] = {}

    for test in tests:
        result = score_query(api_url, test, top_k=top_k)
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        cat = result.get("category", "")
        by_category.setdefault(cat, []).append(result["passed"])

        print(f"{result['query']:<45} {cat:<20} {status}")

        if not result["passed"] and result.get("top_results"):
            # Show what we got instead
            top = result["top_results"][0]
            print(f"    → Top result: {top.get('url', 'N/A')[:70]}  checks={top.get('checks')}")

        if result.get("error"):
            print(f"    → Error: {result['error']}")

        if result["passed"]:
            passed += 1

    total = len(tests)
    print("\n" + "=" * 80)
    print(f"OVERALL: {passed}/{total} passed ({100*passed//total}%)\n")

    print("By category:")
    for cat, results in sorted(by_category.items()):
        cat_pass = sum(results)
        print(f"  {cat:<25} {cat_pass}/{len(results)}")

    return passed, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HS Aalen search engine")
    parser.add_argument("--url", default="http://localhost:8000", help="Search API base URL")
    parser.add_argument("--top-k", type=int, default=3, help="Check top-N results per query")
    args = parser.parse_args()

    passed, total = run_eval(args.url, args.top_k)
    sys.exit(0 if passed == total else 1)
