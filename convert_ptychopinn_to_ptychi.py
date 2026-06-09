#!/usr/bin/env python
"""Convert PtychoPINN .npz data into pty-chi/Ptychodus-style HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def convert_npz_to_ptychi(
    npz_path: Path,
    output_dir: Path,
    pixel_size_m: float = 1.0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    data = np.load(npz_path)
    required = {"diff3d", "probeGuess", "xcoords", "ycoords"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"{npz_path} is missing required keys: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    dp_path = output_dir / f"{stem}_ptychodus_dp.hdf5"
    para_path = output_dir / f"{stem}_ptychodus_para.hdf5"

    if not overwrite and (dp_path.exists() or para_path.exists()):
        raise FileExistsError(
            f"Output exists for {stem}. Pass --overwrite to replace existing files."
        )

    diff3d = data["diff3d"]
    probe = data["probeGuess"]
    xcoords_px = data["xcoords"]
    ycoords_px = data["ycoords"]

    if diff3d.ndim != 3:
        raise ValueError(f"diff3d must be 3D (n, h, w), got {diff3d.shape}")
    if probe.ndim != 2:
        raise ValueError(f"probeGuess must be 2D (h, w), got {probe.shape}")
    if len(xcoords_px) != diff3d.shape[0] or len(ycoords_px) != diff3d.shape[0]:
        raise ValueError("xcoords/ycoords length must match number of diffraction patterns")

    with h5py.File(dp_path, "w") as f:
        f.create_dataset("dp", data=diff3d, compression="gzip")

    with h5py.File(para_path, "w") as f:
        # pty-chi's test loader turns a 3D probe into shape (1, 1, h, w).
        f.create_dataset("probe", data=probe[np.newaxis, ...])
        f.create_dataset("probe_position_x_m", data=xcoords_px * pixel_size_m)
        f.create_dataset("probe_position_y_m", data=ycoords_px * pixel_size_m)
        obj_group = f.create_group("object")
        obj_group.attrs["pixel_height_m"] = pixel_size_m
        obj_group.attrs["pixel_width_m"] = pixel_size_m
        if "objectGuess" in data:
            obj_group.create_dataset("initial_guess", data=data["objectGuess"])

    return dp_path, para_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="PtychoPINN .npz file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("ptychi_converted"),
        help="Directory for converted HDF5 files",
    )
    parser.add_argument(
        "--pixel-size-m",
        type=float,
        default=1.0,
        help=(
            "Pixel size used to convert pixel coordinates to meters. "
            "Use 1.0 to preserve PtychoPINN's pixel-like coordinates."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dp_path, para_path = convert_npz_to_ptychi(
        args.npz,
        args.output_dir,
        pixel_size_m=args.pixel_size_m,
        overwrite=args.overwrite,
    )
    print(f"Wrote diffraction file: {dp_path}")
    print(f"Wrote parameter file:   {para_path}")


if __name__ == "__main__":
    main()
