#!/usr/bin/env python
"""Evaluate a PtychoPINN inference reconstruction against objectGuess."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from ptychopinn_torch.eval.frc import frc_preprocess_images
    from ptychopinn_torch.eval.eval_metrics import FSC
except ImportError:
    frc_preprocess_images = None
    FSC = None


def center_crop(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0 = max((arr.shape[-2] - shape[0]) // 2, 0)
    x0 = max((arr.shape[-1] - shape[1]) // 2, 0)
    return arr[..., y0 : y0 + shape[0], x0 : x0 + shape[1]]


def common_center_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = (min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1]))
    return center_crop(a, shape), center_crop(b, shape)


def finite_bbox(arr: np.ndarray) -> tuple[slice, slice]:
    finite = np.isfinite(arr)
    if np.iscomplexobj(arr):
        finite = np.isfinite(arr.real) & np.isfinite(arr.imag)
    yy, xx = np.where(finite)
    if len(yy) == 0:
        raise ValueError("Reconstruction has no finite pixels.")
    return slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1)


def common_square_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side = min(a.shape[-2], a.shape[-1], b.shape[-2], b.shape[-1])
    return center_crop(a, (side, side)), center_crop(b, (side, side))


def align_complex_scale(reference: np.ndarray, recon: np.ndarray) -> np.ndarray:
    finite = np.isfinite(reference) & np.isfinite(recon)
    if np.iscomplexobj(reference):
        finite &= np.isfinite(reference.real) & np.isfinite(reference.imag)
    if np.iscomplexobj(recon):
        finite &= np.isfinite(recon.real) & np.isfinite(recon.imag)
    if not np.any(finite):
        return recon
    reference_valid = reference[finite]
    recon_valid = recon[finite]
    denom = np.vdot(recon_valid, recon_valid)
    if np.abs(denom) < 1e-12:
        return recon
    return (np.vdot(recon_valid, reference_valid) / denom) * recon


def phase_error(reference: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.angle(recon) - np.angle(reference))))


def load_reference(recon_npz: Path, dataset_dir_override: Path | None) -> tuple[np.ndarray, Path]:
    result = np.load(recon_npz, allow_pickle=True)
    if dataset_dir_override is None:
        if "dataset_dir" not in result.files:
            raise KeyError("No dataset_dir key in reconstruction file; pass --dataset-dir.")
        dataset_dir = Path(str(result["dataset_dir"]))
    else:
        dataset_dir = dataset_dir_override

    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No dataset .npz files found in {dataset_dir}")
    if len(npz_files) > 1:
        print(f"Warning: multiple .npz files found in {dataset_dir}; using {npz_files[0].name}.")

    with np.load(npz_files[0], allow_pickle=True) as dataset:
        if "objectGuess" not in dataset.files:
            raise KeyError(f"{npz_files[0]} does not contain objectGuess.")
        reference = dataset["objectGuess"]
    return reference, npz_files[0]


def ptychopinn_frc_auc(
    reference: np.ndarray,
    recon: np.ndarray,
    cutoff: float,
    align: bool,
) -> float:
    if frc_preprocess_images is None or FSC is None:
        raise RuntimeError("PtychoPINN FRC functions could not be imported.")

    reference_sq, recon_sq = common_square_crop(reference, recon)
    reference_sq = np.nan_to_num(reference_sq)
    recon_sq = np.nan_to_num(recon_sq)
    aligned_ref, aligned_recon = frc_preprocess_images(
        reference_sq,
        recon_sq,
        image_prop="complex",
        verbose=False,
        align=align,
    )
    fr_curve, x_fr, _t_curve, _x_t = FSC(aligned_ref, aligned_recon)
    stop = np.where(x_fr - cutoff > 0)[0]
    stop_idx = int(stop[0]) if len(stop) else len(fr_curve)
    return float(np.sum(fr_curve[:stop_idx]) / max(stop_idx, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recon_npz", type=Path, help="Inference output .npz.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--frc-cutoff", type=float, default=0.5)
    parser.add_argument("--frc-align", action="store_true")
    parser.add_argument(
        "--amp-mask-threshold",
        type=float,
        default=0.05,
        help="Phase metrics use pixels where reference amplitude exceeds this fraction of max.",
    )
    args = parser.parse_args()

    with np.load(args.recon_npz, allow_pickle=True) as result:
        recon = result["object"]

    reference, reference_path = load_reference(args.recon_npz, args.dataset_dir)
    reference, recon = common_center_crop(reference, recon)
    y_slice, x_slice = finite_bbox(recon)
    reference = reference[y_slice, x_slice]
    recon = recon[y_slice, x_slice]
    recon = align_complex_scale(reference, recon)

    amp_ref = np.abs(reference)
    amp_rec = np.abs(recon)
    finite = np.isfinite(amp_ref) & np.isfinite(amp_rec)
    if np.iscomplexobj(reference):
        finite &= np.isfinite(reference.real) & np.isfinite(reference.imag)
    if np.iscomplexobj(recon):
        finite &= np.isfinite(recon.real) & np.isfinite(recon.imag)
    amp_range = max(float(amp_ref.max() - amp_ref.min()), 1e-12)
    amp_rmse = float(np.sqrt(np.mean((amp_rec[finite] - amp_ref[finite]) ** 2)))
    amp_nrmse = amp_rmse / amp_range
    amp_mae = float(np.mean(np.abs(amp_rec[finite] - amp_ref[finite])))

    mask = finite & (amp_ref > args.amp_mask_threshold * amp_ref.max())
    ph_err = phase_error(reference, recon)
    phase_mae_rad = float(np.mean(np.abs(ph_err[mask])))
    phase_rmse_rad = float(np.sqrt(np.mean(ph_err[mask] ** 2)))

    complex_nrmse = float(
        np.linalg.norm((recon[finite] - reference[finite]).ravel())
        / max(np.linalg.norm(reference[finite].ravel()), 1e-12)
    )
    frc_auc = ptychopinn_frc_auc(
        reference,
        recon,
        cutoff=args.frc_cutoff,
        align=args.frc_align,
    )

    print(f"reconstruction: {args.recon_npz}")
    print(f"reference:      {reference_path}:objectGuess")
    print(f"comparison shape: {reference.shape}")
    print(f"complex_nrmse:      {complex_nrmse:.6g}  lower is better")
    print(f"amplitude_mae:      {amp_mae:.6g}  lower is better")
    print(f"amplitude_nrmse:    {amp_nrmse:.6g}  lower is better")
    print(f"phase_mae_rad:      {phase_mae_rad:.6g}  lower is better")
    print(f"phase_rmse_rad:     {phase_rmse_rad:.6g}  lower is better")
    print(f"ptychopinn_frc_auc_0_to_{args.frc_cutoff:g}: {frc_auc:.6g}  higher is better")


if __name__ == "__main__":
    main()
