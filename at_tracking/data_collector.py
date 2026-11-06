"""Collect tracked poses and write them to Parquet.

Run directly to summarize a recording:
    python -m at_tracking.data_collector tracking.parquet
"""

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRACKING_COLUMNS = ["timestamp", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]


class DataCollector:

    def __init__(self, metadata=None):
        self.rows = []
        self.metadata = dict(metadata or {})

    def update(self, timestamp, name, x, y, z, qx, qy, qz, qw):
        self.rows.append([timestamp, name, x, y, z, qx, qy, qz, qw])

    def dump(self, filename="output.parquet", metadata=None):
        """Write long-form tracker poses with JSON metadata in the Parquet footer."""
        df = pd.DataFrame(self.rows, columns=TRACKING_COLUMNS)
        table = pa.Table.from_pandas(df, preserve_index=False)
        file_metadata = dict(self.metadata)
        file_metadata.update(metadata or {})
        file_metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        file_metadata.setdefault("timestamp_clock", "Unix wall clock from time.time()")
        file_metadata.setdefault("timestamp_unit", "seconds")
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[b"dataset_metadata"] = json.dumps(
            file_metadata, sort_keys=True, default=str
        ).encode("utf-8")
        table = table.replace_schema_metadata(schema_metadata)
        pq.write_table(table, filename)


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
