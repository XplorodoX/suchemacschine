import json
import re

INPUT_FILE = "/home/flo/suchemacschine/data.jsonl"
OUTPUT_FILE = "/home/flo/suchemacschine/data_cleaned.jsonl"

# Patterns to remove
PATTERNS_TO_REMOVE = [
    # The AI Search disclaimer at the end
    r"AI Suche Hallo, ich bin deine AI Suche\. Frage mich etwas!.*$",
    # The "Inhalt Ausblenden" summary at the beginning
    r"^Inhalt Ausblenden.*?Inhalt einblenden\s*",
    # Social sharing boilerplate
    r"Teilen Schließen Teilen Facebook LinkedIn Email Url Kopieren Link in Zwischenablage kopiert",
    # Specific common navigation fragments
    r"Häufig gesucht: Künstliche Intelligenz & Informatik Bachelor Artificial Intelligence & Data Science",
]


def clean_text(text):
    for pattern in PATTERNS_TO_REMOVE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print(f"Cleaning {INPUT_FILE}...")
    cleaned_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                record = json.loads(line)
                original_text = record.get("content", "")
                cleaned_text = clean_text(original_text)

                record["content"] = cleaned_text
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                cleaned_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"Cleaned {cleaned_count} records. Saved to {OUTPUT_FILE}")

    # Overwrite the original with the cleaned version if needed, or just tell the user
    # For safety, I'll keep them separate for now but suggest replacing if it looks good.


if __name__ == "__main__":
    main()
