#!/usr/bin/env python3
"""Rebuild the evaluated GOOSE-Ex identities from an author's original download.

Calls the unchanged inventory function. No images, labels, predictions or new
annotations are generated, and nothing is downloaded by this command.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goose-root", type=Path, required=True,
                    help="Extracted images/, labels/ and _archive_metadata/ tree")
    ap.add_argument("--expected", type=Path, required=True,
                    help="Original real_images.json supplied in the numerical deposit")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), "Use a new output directory"
    repo = Path(__file__).resolve().parents[1]
    source = repo / "workspace/strengthening_2026_10_18/goose_ex/code/prepare.py"
    spec = importlib.util.spec_from_file_location("original_goose_inventory", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GOOSE = args.goose_root.resolve()
    module.ROOT = module.GOOSE.parent
    module.OUT = args.out.resolve()
    receipt = module.inventory()
    got = json.loads((module.OUT / "manifests/real_images.json").read_text())
    expected = json.loads(args.expected.read_text())
    key = lambda r: (r["split"], r["id"], r["sequence"])
    assert sorted(map(key, got)) == sorted(map(key, expected)), "Image identities differ"
    selected = sorted((r for r in got if r["split"] == "val"), key=key)
    assert len(selected) == 192 and len({r["sequence"] for r in selected}) == 10
    with (module.OUT / "evaluated_validation_images.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sequence", "rgb", "label"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in selected)
    receipt.update(identities_match_original=True, evaluated_images=len(selected),
                   image_pixels_read=False, new_annotations=False, downloaded=False)
    (module.OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
