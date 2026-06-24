# PtychoPINN Inference Modeling Summary

## Goal

Model PtychoPINN system efficiency by separating:

1. I/O latency
2. Neural-network inference latency

Model load time is treated as either a fixed constant, approximately 1 second on recent runs, or excluded from dataset-scaling analysis.

## Inference Path

The reported `Total inference time` comes from:

```python
model.forward_predict(x, positions, probe, in_scale)
```

It includes:

- input scaling
- the autoencoder neural network
- amplitude/phase to complex object conversion

It does not include:

- physics forward model
- FFT / diffraction simulation
- assembly

`Total assembly time` is post-processing that stitches predicted patches back into a reconstruction canvas.

## Dataset Quantities

For the released PtychoPINN inference models used here:

```text
input per grouped sample = (4, 64, 64)
C = 4
N = 64
object_big = true
cbam_encoder = true
```

Definitions:

```text
R(D) = raw input image count
V(D) = valid scan points after coordinate bounds
S(D) = grouped samples actually used by the NN
```

Empirical relationship for the current real datasets:

```text
S ≈ 0.78R
V ≈ 0.78R
raw / grouped ≈ 1.28
(V + S) / R ≈ 1.56
```

Strictly, `V` and `S` depend on coordinate geometry, bounds, and neighbor grouping, not only on raw image count.

## NN FLOPs

For the current fixed network:

```text
per grouped sample ≈ 835,149,312 MACs
per grouped sample ≈ 1.670 GFLOPs   # using 1 MAC = 2 FLOPs
```

Thus:

```text
F_total(D) = S(D) * 1.670 GFLOPs
```

If resolution changes, per-sample FLOPs changes:

```text
N=64:  1.670 GFLOPs/sample
N=128: 3.774 GFLOPs/sample
N=256: 12.002 GFLOPs/sample
```

## Inference Latency Model

Do not directly use:

```text
time = FLOPs / GPU peak FLOP/s
```

Small datasets are dominated by fixed overhead and batch-level effects. Use:

```text
T_infer(D, B, G) ≈ A(G) + K(B,G) * ceil(S(D) / B)
```

Where:

```text
B = batch size
G = GPU
A(G) = fixed overhead
K(B,G) = per-batch inference time on GPU G
ceil(S/B) = number of NN batches
```

A more detailed version is:

```text
T_infer(D,B,G,N)
≈ A(G)
 + ceil(S(D)/B) * T_batch_overhead(G)
 + S(D) * F_sample(N) / P_eff(B,G,N)
```

For fitting, use:

```text
T_infer ≈ A + K * num_batches
```

Example H100 fit from IC2 and W:

```text
IC2: S=7591,  batches=8,  time≈0.62s
W:   S=20449, batches=20, time≈1.04s

A_H100 ≈ 0.34s
K_H100 ≈ 0.035s/batch
```

## I/O Latency Model

For warm-cache / fast local memmap experiments, model only the two dominant steps:

1. `group_coords`
2. `write image tensors`

Use:

```text
T_IO(D) ≈ T_group(D) + T_write(D)
T_IO(D) ≈ α · V(D) + β · S(D)
```

Empirical H100 `/tmp` estimates:

```text
α ≈ 70 μs / valid point
β ≈ 40 μs / grouped sample
```

If only raw image count `R` is known:

```text
V ≈ 0.78R
S ≈ 0.78R
```

Then:

```text
T_IO_seconds
≈ 70e-6 * 0.78R + 40e-6 * 0.78R
≈ 86e-6 * R
```

`α` and `β` are machine-dependent and should be calibrated per machine.

## Machine Calibration

On each machine, run the synthetic sweep and collect:

- `group_coords_time`
- `write_image_tensors_time`
- `Total inference time`
- power CSV

Fit:

```text
α = group_coords_time / V
β = write_image_tensors_time / S
```

and:

```text
T_infer ≈ A + K * ceil(S/B)
```

## Synthetic Dataset Experiment

Because reconstruction quality is irrelevant for system modeling, synthetic inputs are sufficient.

Raw image counts:

```text
1000, 2000, 4000, 8000, 12000, 16000, 20000, 26000
```

Scripts:

```text
PtychoPINN/script_modeling/create_synthetic_inference_dataset.py
PtychoPINN/script_modeling/create_synthetic_inference_sweep.sh
PtychoPINN/script_modeling/run_inference_experiments.py
PtychoPINN/script_modeling/run_inference_experiments.sh
```

Default output directory:

```text
PtychoPINN/modeling_exp/
```

Synthetic datasets:

```text
PtychoPINN/synthetic_inputs/
```

Synthetic datasets are ignored by git. Results are intended to be git-addable.

## Run Commands

Generate synthetic datasets:

```bash
cd /home/zhong.zheng/PtychoPINN
./script_modeling/create_synthetic_inference_sweep.sh
```

Run modeling inference experiments:

```bash
./script_modeling/run_inference_experiments.sh
```

Platform wrappers:

```bash
# ARM/Grace + NVIDIA GPU
./script_modeling/run_inference_experiments_arm.sh

# ARM CPU-only
DEVICE=cpu VENDOR=auto DEVICES="" ./script_modeling/run_inference_experiments_arm.sh

# AMD ROCm GPU
./script_modeling/run_inference_experiments_amd.sh

# Intel XPU GPU
./script_modeling/run_inference_experiments_intel.sh
```

```bash
./script_modeling/run_inference_experiments.sh
```

Defaults:

```text
batch size = 1024
model key = PS_W
dataset profile = enabled
power logging = enabled
output = PtychoPINN/modeling_exp/
```

Summary CSV fields include:

```text
dataset
batch_size
dataset_length
profile_group_coords_s
profile_write_image_tensors_s
profile_load_diff3d_s
profile_dataset_init_s
model_load_time_s
data_load_time_s
inference_time_s
assembly_time_s
power_csv
log_file
```
