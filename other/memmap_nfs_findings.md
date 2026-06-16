# TensorDict Memmap and NFS Loading Findings

## Summary

The large data loading time difference between the H100 and GH200 systems was caused mainly by where the temporary TensorDict `_memmap` cache was written, not by the original `.npz` dataset read or GPU compute speed.

Both machines read the original dataset from the same NFS-mounted home filesystem:

```text
/home -> rhino-01-jlse:/vol/ft_home, type nfs
```

However, when `_memmap` was created under the repo or `power_experiments` directory, it was also written to `/home`, so it used NFS. On H100 this made TensorDict memmap writes and reloads much slower.

Using local node storage for `_memmap` fixed the slowdown:

```bash
MEMMAP_ROOT=/tmp/ptycho_memmap ./scripts/run_inference_experiments.sh
```

For GH200/ARM:

```bash
MEMMAP_ROOT=/tmp/ptycho_memmap ./scripts/run_inference_experiments_arm.sh
```

## Key Measurements

For dataset `IC2`, batch size `128`, with `_memmap` on `/home` NFS on H100:

```text
memory_map_data dataset 0 write image tensors: 6.336776 s
TensorDict.load_memmap: 2.230394 s
PtychoDataset __init__ total: 9.928156 s
```

With `_memmap` moved to local `/tmp` on H100:

```text
memory_map_data dataset 0 write image tensors: 0.308057 s
TensorDict.load_memmap: 0.014617 s
PtychoDataset __init__ total: 0.923235 s
```

This shows the bottleneck was the generated memmap cache on NFS.

## What `_memmap` Is

The original `.npz` dataset contains arrays such as:

```text
diff3d
xcoords
ycoords
probeGuess
objectGuess
```

The dataloader preprocesses these into model-ready tensors:

```text
filter coordinates
group neighboring diffraction patterns
build coords_relative / coords_global / nn_indices
normalize diffraction images
store scaling constants
reshape images for model input
```

Those processed tensors are stored in a TensorDict `_memmap` cache so the DataLoader can index batches efficiently.

The important distinction is:

```text
Original .npz read: still from /home NFS
Temporary _memmap write/load: should be on /tmp local disk for speed
```

## Why GH200 Was Still Fast on NFS

Even though GH200 also used the same `/home` NFS mount, its NFS client path handled this TensorDict memmap workload much faster than H100. Possible reasons include different CPU architecture, kernel/NFS behavior, page-cache behavior, network path, or node contention.

For a fair accelerator comparison, the benchmark should avoid measuring this machine-specific NFS behavior. Put `_memmap` on local disk on all machines.

## Recommendation

For benchmark sweeps, use local `/tmp` and delete `_memmap` after each batch, so each batch size starts from the same condition:

```bash
MEMMAP_ROOT=/tmp/ptycho_memmap ./scripts/run_inference_experiments.sh
```

For repeated real-world use, keep the memmap during the job, but still keep it on local disk:

```bash
MEMMAP_ROOT=/tmp/ptycho_memmap ./scripts/run_inference_experiments.sh --keep-memmap
```

Keeping `_memmap` on NFS avoids recreating it, but `TensorDict.load_memmap` can still be slow on H100. The best setup is to create and reuse `_memmap` on local node storage or node-local scratch.

## Useful Checks

Check where the dataset lives:

```bash
df -h data/IC2
df -T data/IC2
```

Check local temporary storage:

```bash
df -h /tmp
df -T /tmp
du -sh /tmp/ptycho_memmap 2>/dev/null
```

Enable detailed dataloader timing:

```bash
PTYCHO_DATASET_PROFILE=1 MEMMAP_ROOT=/tmp/ptycho_memmap ./scripts/run_inference_experiments.sh --test --datasets IC2 --batch-sizes 128
```

Important profile lines to compare:

```text
memory_map_data dataset 0 write image tensors
TensorDict.load_memmap
PtychoDataset __init__ total
```
