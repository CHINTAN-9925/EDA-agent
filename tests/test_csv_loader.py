"""Tests for safe in-memory CSV loading."""

import pytest

from utils.csv_loader import CSVLoadError, load_csv


def test_csv_loader_reads_valid_bytes() -> None:
    loaded = load_csv(b"name,value\na,1\nb,2\n", "sample.csv")
    assert loaded.dataframe.shape == (2, 2)
    assert loaded.encoding == "utf-8-sig"


@pytest.mark.parametrize(
    ("data", "filename", "message"),
    [
        (b"", "empty.csv", "empty"),
        (b"name\n", "headers.csv", "zero data rows"),
        (b"name\na\n", "sample.txt", "Only .csv"),
    ],
)
def test_csv_loader_rejects_invalid_uploads(data: bytes, filename: str, message: str) -> None:
    with pytest.raises(CSVLoadError, match=message):
        load_csv(data, filename)
