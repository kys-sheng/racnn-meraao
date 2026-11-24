"""Download the released weights and data from Hugging Face into weights/ and data/.

    python scripts/download.py                      # everything (~1.4 GB)
    python scripts/download.py --analysis bumphunt  # bump hunt only (~0.27 GB)
    python scripts/download.py --no-train           # skip the main train/val splits (~0.75 GB; enough for the example notebooks)
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "kys-sheng/racnn-meraao"   # same name for the model repo and the dataset repo
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--analysis", choices=("bumphunt", "monojet", "all"), default="all")
p.add_argument("--no-train", action="store_true", help="skip train.h5 / val.h5 of the main datasets")
p.add_argument("--no-data", action="store_true", help="weights only")
args = p.parse_args()

analyses = ("bumphunt", "monojet") if args.analysis == "all" else (args.analysis,)
common = ["*.json", "*.md"]

snapshot_download(REPO_ID, repo_type="model", local_dir=os.path.join(ROOT, "weights"),
                  allow_patterns=common + [f"{a}/*" for a in analyses])

if not args.no_data:
    ignore = [f"{a}/train.h5" for a in analyses] + [f"{a}/val.h5" for a in analyses] if args.no_train else None
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=os.path.join(ROOT, "data"),
                      allow_patterns=common + ["figure_data/*"] + [f"{a}/*" for a in analyses], ignore_patterns=ignore)

print("weights ->", os.path.join(ROOT, "weights"))
if not args.no_data:
    print("data    ->", os.path.join(ROOT, "data"))
