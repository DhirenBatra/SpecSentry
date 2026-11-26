"""Download and verify the exact PROMISE_exp snapshot used by SpecSentry."""

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parent
RAW_PATH = DATA_DIR / "raw" / "PROMISE_exp.csv"

SOURCE_REPOSITORY = "https://github.com/AleksandarMitrevski/se-requirements-classification"
SOURCE_COMMIT = "1f5dc4501a1956f21011083594d925bac49f198c"
SOURCE_PATH = "1-exploratory-analysis/data/PROMISE_exp.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/AleksandarMitrevski/"
    f"se-requirements-classification/{SOURCE_COMMIT}/{SOURCE_PATH}"
)
SOURCE_SHA256 = "1b2cc33743588748ced7abd74a527e92f6f6d0dbe5342692cd2aa101e864f34f"


def verify_source(data: bytes) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"Expected SHA-256 {SOURCE_SHA256}, got {actual}.")


def download() -> bool:
    """Return True for a new download or False for an existing verified file."""
    if RAW_PATH.exists():
        try:
            verify_source(RAW_PATH.read_bytes())
        except ValueError as error:
            raise ValueError(f"Refusing to overwrite {RAW_PATH}. {error}") from error
        return False

    with urlopen(SOURCE_URL, timeout=30) as response:
        data = response.read()
    verify_source(data)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("xb") as file:
        file.write(data)
    return True


def main() -> None:
    try:
        downloaded = download()
    except (OSError, ValueError) as error:
        sys.exit(f"PROMISE_exp download failed for {SOURCE_URL}\n{error}")

    action = "Downloaded and verified" if downloaded else "Verified existing"
    print(f"{action} PROMISE_exp at {RAW_PATH}")


if __name__ == "__main__":
    main()
