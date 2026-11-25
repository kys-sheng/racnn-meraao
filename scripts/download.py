"""Download the released datasets from Hugging Face into data/ (the trained weights are already in weights/).

    python scripts/download.py                      # everything (~1.3 GB)
    python scripts/download.py --analysis bumphunt  # bump hunt only (~0.22 GB)
    python scripts/download.py --no-train           # skip the main train/val splits (~0.7 GB; enough for the example notebooks)
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "kys-sheng/racnn-meraao"   # Hugging Face dataset repo
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--analysis", choices=("bumphunt", "monojet", "all"), default="all")
p.add_argument("--no-train", action="store_true", help="skip train.h5 / val.h5 of the main datasets")
args = p.parse_args()

analyses = ("bumphunt", "monojet") if args.analysis == "all" else (args.analysis,)
ignore = [f"{a}/{split}.h5" for a in analyses for split in ("train", "val")] if args.no_train else None

snapshot_download(REPO_ID, repo_type="dataset", local_dir=os.path.join(ROOT, "data"),
                  allow_patterns=["*.md", "figure_data/*"] + [f"{a}/*" for a in analyses], ignore_patterns=ignore)

print("data ->", os.path.join(ROOT, "data"))
