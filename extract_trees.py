#!/usr/bin/env python3
"""
Split a LAS/LAZ point cloud into one file per instance, using an extra dimension
named 'instance_pred'.

Usage:
  python split_instances.py input.las --outdir instances_out --skip -1 4294967295 --compress

Notes:
- If your instance_pred is stored as uint32, "no-instance" is often 4294967295.
- This script keeps the original header and point format, and subsets points safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import laspy


def split_by_instance(
    in_path: Path,
    out_dir: Path,
    instance_field: str = "instance_pred",
    skip_values: list[int] | None = None,
    min_points: int = 1,
    compress: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    las = laspy.read(str(in_path))

    # Check that the attribute exists
    dims = set(las.point_format.dimension_names)
    if instance_field not in dims:
        raise ValueError(
            f"'{instance_field}' not found in point dimensions.\n"
            f"Available dimensions: {sorted(dims)}"
        )

    inst = np.asarray(getattr(las, instance_field))
    if inst.ndim != 1:
        inst = inst.reshape(-1)

    # Determine which instance IDs to process
    unique_ids = np.unique(inst)

    if skip_values:
        skip_set = set(int(v) for v in skip_values)
        unique_ids = np.array([i for i in unique_ids if int(i) not in skip_set], dtype=unique_ids.dtype)

    # Write one file per instance
    written = 0
    for iid in unique_ids:
        mask = inst == iid
        n = int(mask.sum())
        if n < min_points:
            continue

        sub = las[mask]  # subsets while preserving point format + extra dims

        suffix = ".laz" if compress else ".las"
        out_path = out_dir / f"{in_path.stem}__instance_{int(iid)}__n{n}{suffix}"

        # Write (optionally compressed if lazrs/laszip backend is available)
        sub.write(str(out_path))
        written += 1

    print(f"Done. Wrote {written} instance files to: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Split LAS/LAZ by instance_pred into separate files.")
    p.add_argument("input", type=Path, help="Input .las or .laz")
    p.add_argument("--outdir", type=Path, default=Path("instances_out"), help="Output directory")
    p.add_argument("--field", type=str, default="instance_pred", help="Instance field name (default: instance_pred)")
    p.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="Instance IDs to skip (e.g. -1 4294967295)",
    )
    p.add_argument("--min-points", type=int, default=1, help="Skip instances with fewer than this many points")
    p.add_argument(
        "--compress",
        action="store_true",
        help="Write LAZ instead of LAS (requires a compression backend, e.g. lazrs)",
    )
    args = p.parse_args()

    split_by_instance(
        in_path=args.input,
        out_dir=args.outdir,
        instance_field=args.field,
        skip_values=args.skip,
        min_points=args.min_points,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
