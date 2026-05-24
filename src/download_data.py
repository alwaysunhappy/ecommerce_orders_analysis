from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import EXPECTED_FILES, RAW_DIR, ensure_directories

DATASET_HANDLE = "olistbr/brazilian-ecommerce"


def download_olist_dataset(raw_dir: Path = RAW_DIR) -> list[Path]:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Package kagglehub is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    ensure_directories()
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    copied_files: list[Path] = []

    for filename in EXPECTED_FILES.values():
        matches = list(dataset_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Expected file was not found in downloaded dataset: {filename}")

        destination = raw_dir / filename
        shutil.copy2(matches[0], destination)
        copied_files.append(destination)
        print(f"Saved {destination}")

    return copied_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Olist public e-commerce dataset from Kaggle.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory for raw CSV files.")
    args = parser.parse_args()
    download_olist_dataset(args.raw_dir)


if __name__ == "__main__":
    main()
