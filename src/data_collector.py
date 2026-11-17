"""Collect tracked poses and write them to Parquet.

Run directly to summarize a recording:
    python -m src.data_collector tracking.parquet
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRACKING_COLUMNS = ["timestamp", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]
METADATA_KEY = b"dataset_metadata"


def write_tracking_parquet(filename, frames, metadata):
    """Write tracking rows with JSON metadata in the Parquet footer.

    Args:
        filename: Output path.
        frames:   DataFrame with TRACKING_COLUMNS.
        metadata: dict stored as JSON under METADATA_KEY.
    """
    table = pa.Table.from_pandas(frames, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[METADATA_KEY] = json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(schema_metadata), filename)


def read_tracking_parquet(filename):
    """Return a recording's rows as a DataFrame, and its metadata dict."""
    table = pq.read_table(filename)
    return table.to_pandas(), json.loads(table.schema.metadata[METADATA_KEY])


class DataCollector:

    def __init__(self, metadata=None):
        """Start with no rows and optional metadata for the Parquet footer."""
        self.rows = []
        self.metadata = dict(metadata or {})

    def update(self, timestamp, name, x, y, z, qx, qy, qz, qw):
        """Add one pose row."""
        self.rows.append([timestamp, name, x, y, z, qx, qy, qz, qw])

    def dump(self, filename="output.parquet", metadata=None):
        """Write long-form tracker poses with JSON metadata in the Parquet footer."""
        file_metadata = dict(self.metadata)
        file_metadata.update(metadata or {})
        file_metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        file_metadata.setdefault("timestamp_clock", "Unix wall clock from time.time()")
        file_metadata.setdefault("timestamp_unit", "seconds")
        write_tracking_parquet(
            filename, pd.DataFrame(self.rows, columns=TRACKING_COLUMNS), file_metadata
        )


def summarize(filename):
    """Print how often each object was seen in a recording."""
    df = pq.read_table(filename).to_pandas()
    if df.empty:
        print(f"{filename} has no rows")
        return

    frames = df["timestamp"].nunique()
    for name, count in df.groupby("name").size().items():
        print(f"{name}: seen in {count} of {frames} tracked frames ({count / frames:.1%})")

    duration = df["timestamp"].max() - df["timestamp"].min()
    if duration > 0:
        print(f"length was {duration:.2f} seconds, or {frames / duration:.2f} tracked fps")


def main():
    parser = argparse.ArgumentParser(description="Summarize a tracking Parquet file.")
    parser.add_argument("filename")
    summarize(parser.parse_args().filename)


if __name__ == "__main__":
    main()
