import os
import csv
from typing import List, Dict

"""
Print and export selected charters from one or more CSV files.

Configure CSV_FILES and TARGET_NUMMERS below. If TARGET_NUMMERS is empty, all records are exported.
Output is written to csv/selected_urkunden.csv.
"""

# CONFIG: which CSV files to scan
CSV_FILES: List[str] = [
    # os.path.join("csv", "FII_1-170.csv"),
    os.path.join("csv", "FII_171-235.csv"),
    # os.path.join("csv", "FII_236-299.csv"),
    # os.path.join("csv", "FII_300-363.csv"),
    # os.path.join("csv", "FII_364-426.csv"),
    # os.path.join("csv", "FII_427-657.csv"),
    # os.path.join("csv", "FII_658-929.csv"),
    # os.path.join("csv", "FII_930-1199.csv"),
    # os.path.join("csv", "FII_1200-1463.csv"),
    # os.path.join("csv", "FII_1464-1721.csv"),
]

# CONFIG: which specific charters to export (leave empty to export all)
TARGET_NUMMERS: List[str] = [
    # "178",
    # "1464",
]

OUTPUT_CSV = os.path.join("csv", "selected_urkunden.csv")

USE_COLOR = True

# ANSI colors
COLOR_NUMMER = "\033[96m"   # cyan
COLOR_SUMMARY = "\033[92m"  # green
COLOR_LATIN = "\033[93m"    # yellow
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"


def normalize_nummer(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    # strip possible trailing dot like "1464." to "1464"
    if v.endswith('.'):
        v = v[:-1]
    return v


def read_records(csv_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not os.path.exists(csv_path):
        return records
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nummer = row.get("nummer") or row.get("id") or row.get("Nr")
            nummer = normalize_nummer(nummer)
            summary = (row.get("summary_de") or "").strip()
            latin = (row.get("latin_text") or row.get("latin") or "").strip()

            record = {
                "nummer": nummer,
                "summary_de": summary,
                "latin_text": latin,
                "_source": os.path.basename(csv_path),
            }
            records.append(record)
    return records


def filter_records(records: List[Dict[str, str]], targets: List[str]) -> List[Dict[str, str]]:
    if not targets:
        return records
    target_set = {normalize_nummer(t) for t in targets}
    return [r for r in records if normalize_nummer(r.get("nummer")) in target_set]


def print_records(records: List[Dict[str, str]]):
    for r in records:
        if USE_COLOR:
            print(f"\n{COLOR_DIM}---{COLOR_RESET} {COLOR_NUMMER}Urkunde {r['nummer']}{COLOR_RESET} {COLOR_DIM}(aus {r['_source']}){COLOR_RESET} {COLOR_DIM}---{COLOR_RESET}")
            print(f"{COLOR_SUMMARY}Summary:{COLOR_RESET} {r['summary_de']}")
            print(f"{COLOR_LATIN}Latin:{COLOR_RESET} {r['latin_text']}")
        else:
            print(f"\n--- Urkunde {r['nummer']} (aus {r['_source']}) ---")
            print(f"Summary: {r['summary_de']}")
            print(f"Latin: {r['latin_text']}")


def write_output_csv(records: List[Dict[str, str]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nummer", "summary_de", "latin_text"]) 
        writer.writeheader()
        for r in records:
            writer.writerow({
                "nummer": r.get("nummer", ""),
                "summary_de": r.get("summary_de", ""),
                "latin_text": r.get("latin_text", ""),
            })


def main():
    all_records: List[Dict[str, str]] = []
    for path in CSV_FILES:
        all_records.extend(read_records(path))

    selected = filter_records(all_records, TARGET_NUMMERS)

    if TARGET_NUMMERS:
        order = {normalize_nummer(n): i for i, n in enumerate(TARGET_NUMMERS)}
        selected.sort(key=lambda r: order.get(normalize_nummer(r.get("nummer")), 10**9))
    else:
        # otherwise sort by the charter number numerically when possible
        def sort_key(r: Dict[str, str]):
            n = normalize_nummer(r.get("nummer", ""))
            try:
                return int(n)
            except ValueError:
                return 10**9
        selected.sort(key=sort_key)

    print_records(selected)
    write_output_csv(selected, OUTPUT_CSV)
    print(f"\nSaved {len(selected)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


