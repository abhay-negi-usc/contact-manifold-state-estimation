"""
exhaustive_sampling_data_processing.py

Utilities for loading and analyzing exhaustive SE(3) contact sampling data
produced by contact_sampling_voxel_sdf_exhaustive.py.

Supports:
- HDF5 streaming loads
- chunked analysis
- optional CSV export (streaming, safe for huge datasets)
"""

import os
import h5py
import numpy as np
from typing import Iterator, Dict, Optional


# ----------------------------
# CONFIG
# ----------------------------
CFG = {
    "data_file": "./data_generation/data/cylinder_keyway02/sdf_based_contact_data/exhaustive_samples_voxel_sdf.h5",

    # Streaming chunk size
    "chunk_size": 1_000_000,

    # Analysis thresholds
    "analysis": {
        "max_penetration_thresh": 1e-4,
        "near_contact_thresh": 5e-4,
    },

    # CSV export options
    "csv": {
        "enabled": True,                 # <-- turn on CSV export here
        "filtered_only": False,            # only save contact region
        "output_path": "./data_generation/data/cylinder_keyway02/sdf_based_contact_data/contact_samples.csv",
    },
}


# ----------------------------
# Dataset wrapper
# ----------------------------
class ExhaustiveSamplingDataset:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self._h5: Optional[h5py.File] = None

    def open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError("Dataset not opened")
        return self._h5

    @property
    def length(self) -> int:
        return int(self.h5["t"].shape[0])

    def get_slice(self, sl: slice) -> Dict[str, np.ndarray]:
        return {
            "t": self.h5["t"][sl],
            "r": self.h5["r"][sl],
            "config_sdf": self.h5["config_sdf"][sl],
            "min_separation": self.h5["min_separation"][sl],
            "max_penetration": self.h5["max_penetration"][sl],
            "contact_band": self.h5["contact_band"][sl].astype(bool),
            "contact_valid": self.h5["contact_valid"][sl].astype(bool),
        }

    def iter_chunks(self, chunk_size: int):
        N = self.length
        for i in range(0, N, chunk_size):
            yield self.get_slice(slice(i, min(N, i + chunk_size)))


# ----------------------------
# Filtering logic
# ----------------------------
def contact_filter(chunk: Dict[str, np.ndarray], *, max_pen: float, near: float):
    return (
        chunk["contact_band"]
        & (chunk["max_penetration"] <= max_pen)
        & (np.abs(chunk["config_sdf"]) <= near)
    )


# ----------------------------
# CSV export
# ----------------------------
def stream_to_csv(
    dataset: ExhaustiveSamplingDataset,
    *,
    output_path: str,
    filtered_only: bool,
):
    import csv

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    header = [
        "t1", "t2", "t3",
        "r1", "r2", "r3",
        "config_sdf",
        "min_separation",
        "max_penetration",
        "contact_band",
        "contact_valid",
    ]

    wrote_header = False
    total_written = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        for chunk in dataset.iter_chunks(CFG["chunk_size"]):
            if filtered_only:
                mask = contact_filter(
                    chunk,
                    max_pen=CFG["analysis"]["max_penetration_thresh"],
                    near=CFG["analysis"]["near_contact_thresh"],
                )
                if not mask.any():
                    continue
            else:
                mask = slice(None)

            t = chunk["t"][mask]
            r = chunk["r"][mask]

            rows = np.column_stack([
                t[:, 0], t[:, 1], t[:, 2],
                r[:, 0], r[:, 1], r[:, 2],
                chunk["config_sdf"][mask],
                chunk["min_separation"][mask],
                chunk["max_penetration"][mask],
                chunk["contact_band"][mask].astype(np.uint8),
                chunk["contact_valid"][mask].astype(np.uint8),
            ])

            if not wrote_header:
                writer.writerow(header)
                wrote_header = True

            writer.writerows(rows.tolist())
            total_written += rows.shape[0]

    print(f"[CSV] wrote {total_written:,} rows → {output_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    ds = ExhaustiveSamplingDataset(CFG["data_file"]).open()

    print("[Dataset]")
    print("  file:", ds.path)
    print("  samples:", ds.length)

    if CFG["csv"]["enabled"]:
        print("[CSV] exporting …")
        stream_to_csv(
            ds,
            output_path=CFG["csv"]["output_path"],
            filtered_only=CFG["csv"]["filtered_only"],
        )

    ds.close()


if __name__ == "__main__":
    main()
