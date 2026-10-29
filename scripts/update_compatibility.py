"""Refresh official versioned compatibility facts, or check generated docs offline."""

import argparse
import json
from pathlib import Path

from compatibility.catalog import build_catalog, change_summary, update_readme
from compatibility.sources import GitHubSources

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check README against saved JSON; no network"
    )
    parser.add_argument(
        "--summary", type=Path, help="Write a Markdown change summary for the update PR"
    )
    args = parser.parse_args(argv)
    catalog_path = ROOT / "compatibility.json"
    readme_path = ROOT / "README.md"
    original = readme_path.read_text(encoding="utf-8")
    policy = json.loads((ROOT / "compatibility-policy.json").read_text(encoding="utf-8"))
    if args.check:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog["policy"] != policy or update_readme(original, catalog) != original:
            parser.error(
                "Compatibility catalog / README is stale; run scripts/update_compatibility.py"
            )
        print("Compatibility table is consistent (offline check)")
        return
    # Fetch and parse everything before touching tracked files. HTTP failures and
    # changed table formats fail the run; they never erase a previous good catalog.
    catalog = build_catalog(GitHubSources(), policy)
    readme = update_readme(original, catalog)
    previous = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    if args.summary:
        args.summary.write_text(change_summary(previous, catalog), encoding="utf-8")
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme_path.write_text(readme, encoding="utf-8")
    print(f"Updated {len(catalog['images'])} image entries; no serving tests executed")


if __name__ == "__main__":
    main()
