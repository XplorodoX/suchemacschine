import json
import re
from datetime import datetime
from urllib.parse import urlparse

INPUT_FILE = "/home/flo/suchemacschine/data.jsonl"
OUTPUT_REPORT = "/home/flo/suchemacschine/content_quality_report.json"

MIN_CONTENT_LENGTH = 350
MIN_SECTIONS = 1
MIN_HEADINGS = 1
HIGH_JS_GAIN = 500


def load_records(file_path: str) -> list[dict]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def classify_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if "/aktuelles/news/" in path:
        return "news"
    if "/veranstaltungen/" in path:
        return "event"
    if re.search(r"/studiengaenge|/fakultaet|/hochschule", path):
        return "institution"
    return "page"


def to_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def analyze_record(record: dict) -> dict:
    url = record.get("url", "")
    content = (record.get("content") or "").strip()
    sections = record.get("sections") or []
    headings = record.get("headings") or []
    title = (record.get("title") or "").strip()
    h1 = (record.get("h1") or "").strip()

    static_len = to_int(record.get("static_content_length", len(content)))
    rendered_len = to_int(record.get("rendered_content_length", len(content)))
    gain = to_int(record.get("content_gain_from_js", max(0, rendered_len - static_len)))

    used_js = bool(record.get("used_js_render", False))
    attempted_js = bool(record.get("js_render_attempted", False))

    reasons = []
    severity = "ok"

    if len(content) < MIN_CONTENT_LENGTH:
        reasons.append(f"short_content:{len(content)}")

    if len(sections) < MIN_SECTIONS:
        reasons.append(f"few_sections:{len(sections)}")

    if len(headings) < MIN_HEADINGS:
        reasons.append(f"few_headings:{len(headings)}")

    if not title:
        reasons.append("missing_title")

    if not h1:
        reasons.append("missing_h1")

    if attempted_js and not used_js and len(content) < MIN_CONTENT_LENGTH:
        reasons.append("js_attempted_but_no_gain")

    if gain >= HIGH_JS_GAIN:
        reasons.append(f"high_js_gain:{gain}")

    if reasons:
        severity = "high" if len(reasons) >= 3 or len(content) < 180 else "medium"

    return {
        "url": url,
        "url_type": classify_url(url),
        "severity": severity,
        "content_length": len(content),
        "sections_count": len(sections),
        "headings_count": len(headings),
        "used_js_render": used_js,
        "js_render_attempted": attempted_js,
        "static_content_length": static_len,
        "rendered_content_length": rendered_len,
        "content_gain_from_js": gain,
        "reasons": reasons,
    }


def summarize(analyzed: list[dict]) -> dict:
    total = len(analyzed)
    flagged = [item for item in analyzed if item["severity"] != "ok"]
    high = [item for item in flagged if item["severity"] == "high"]
    medium = [item for item in flagged if item["severity"] == "medium"]

    js_attempted = sum(1 for item in analyzed if item["js_render_attempted"])
    js_used = sum(1 for item in analyzed if item["used_js_render"])
    positive_gain = [item for item in analyzed if item["content_gain_from_js"] > 0]

    by_type = {}
    for item in analyzed:
        key = item["url_type"]
        by_type.setdefault(key, {"count": 0, "flagged": 0})
        by_type[key]["count"] += 1
        if item["severity"] != "ok":
            by_type[key]["flagged"] += 1

    top_issues = sorted(
        flagged,
        key=lambda x: (
            0 if x["severity"] == "high" else 1,
            -len(x["reasons"]),
            x["content_length"],
        ),
    )[:60]

    return {
        "timestamp": datetime.now().isoformat(),
        "thresholds": {
            "min_content_length": MIN_CONTENT_LENGTH,
            "min_sections": MIN_SECTIONS,
            "min_headings": MIN_HEADINGS,
            "high_js_gain": HIGH_JS_GAIN,
        },
        "summary": {
            "total_records": total,
            "flagged_count": len(flagged),
            "high_severity_count": len(high),
            "medium_severity_count": len(medium),
            "js_attempted_count": js_attempted,
            "js_used_count": js_used,
            "positive_js_gain_count": len(positive_gain),
            "avg_js_gain_positive": (
                round(sum(item["content_gain_from_js"] for item in positive_gain) / len(positive_gain), 1)
                if positive_gain
                else 0
            ),
        },
        "flagged_by_url_type": by_type,
        "top_issues": top_issues,
    }


def main():
    records = load_records(INPUT_FILE)
    analyzed = [analyze_record(record) for record in records]
    report = summarize(analyzed)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Content quality audit complete")
    print(f"Records: {report['summary']['total_records']}")
    print(f"Flagged: {report['summary']['flagged_count']}")
    print(f"High severity: {report['summary']['high_severity_count']}")
    print(f"JS attempted: {report['summary']['js_attempted_count']}")
    print(f"JS used: {report['summary']['js_used_count']}")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
