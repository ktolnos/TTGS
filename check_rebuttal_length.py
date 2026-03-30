import re
import sys

def check_rebuttal_lengths(filepath, limit=5000):
    with open(filepath) as f:
        text = f.read()

    sections = re.split(r'^# ', text, flags=re.MULTILINE)
    sections = [s for s in sections if s.strip()]

    all_ok = True
    for section in sections:
        title = section.split('\n', 1)[0].strip()
        length = len(section)
        status = "OK" if length <= limit else "OVER"
        if status == "OVER":
            all_ok = False
        print(f"[{status}] {title}: {length} / {limit} chars")

    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    check_rebuttal_lengths("rebuttals.md")
