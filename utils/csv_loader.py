"""Safe, in-memory CSV loading with useful validation errors."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd


MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class CSVLoadError(ValueError):
    """Raised when an uploaded CSV cannot be safely loaded."""


@dataclass(frozen=True)
class LoadedCSV:
    dataframe: pd.DataFrame
    encoding: str
    warnings: list[str]


def load_csv(data: bytes, filename: str = "upload.csv") -> LoadedCSV:
    """Load CSV bytes using common encodings and reject invalid datasets."""
    if not filename.lower().endswith(".csv"):
        raise CSVLoadError("Only .csv files are supported.")
    if not data:
        raise CSVLoadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise CSVLoadError("The file is larger than the 200 MB upload safety limit.")

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(data), encoding=encoding, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    else:
        raise CSVLoadError("Could not parse the CSV. " + " | ".join(errors[-2:]))

    if frame.shape[0] == 0:
        raise CSVLoadError("The CSV has column headers but contains zero data rows.")
    if frame.shape[1] == 0:
        raise CSVLoadError("The CSV contains no columns.")

    warnings: list[str] = []
    mangled = [str(column) for column in frame.columns if "." in str(column) and str(column).rsplit(".", 1)[-1].isdigit()]
    if mangled:
        warnings.append("Pandas renamed possible duplicate column names: " + ", ".join(mangled[:10]))
    if frame.shape[1] == 1:
        warnings.append("The dataset has only one column; relationship analyses will be limited.")
    if len(frame) > 100_000:
        warnings.append("Large dataset: aggregate statistics use all rows, while charts use a sample of at most 10,000 rows.")
    return LoadedCSV(frame, encoding, warnings)
