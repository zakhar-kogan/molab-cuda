# Porting a Kaggle GPU notebook to Molab

Point an agent at this file: *"Use https://github.com/zakhar-kogan/molab-cuda/blob/master/PORTING.md to adapt this notebook."*

```
Port this Kaggle GPU notebook to molab: <MOLAB_URL> token <TOKEN>
Connect with /marimo-pair. Source: <notebook/repo>

- Toolkit: github.com/zakhar-kogan/molab-cuda — molab has driver+torch, no nvcc/cmake.
- CayleyPy solver? Use zakhar-kogan/MultiGPUBeamSearch @ fix/single-gpu-and-build-portability.
- Never edit the .py; drive the kernel via marimo._code_mode.
- Cache in /marimo/storage/: toolkit, PIP_CACHE_DIR, kagglehub, repo, build.
  It persists across sandboxes but LOSSILY — validate caches by content, delete partials.
- Detect GPU count/arch. Kaggle code assumes 2 GPUs and sm_75/86.
- Run long jobs detached; marimo kills cells on disconnect. Sandboxes die.
- Not done until it passes the notebook's own correctness gate.

Done: runs end-to-end verified, warm restart with zero compiles/downloads.
```

## Why each line is there

Every rule above cost a debugging round on a real port.

- **No toolkit.** Fails as a bare `FileNotFoundError: 'cmake'`. See the README for
  the wheel-assembly details (package renames, CCCL version skew, missing sonames).
- **Lossy storage.** Molab storage survives a sandbox but drops symlinks, `.git`
  and large files. Tools then see `.git` or a populated cache dir and skip the
  repair, failing minutes later somewhere unrelated.
- **2-GPU assumptions.** Kaggle-derived code hardcodes two ranks. Symptom: the
  solver finishes correctly, then crashes in reporting.
- **Detached jobs.** Marimo ties cell execution to the client connection; a
  dropped websocket interrupts a running build.
- **Correctness gate.** Use whatever the notebook already has (exact replay,
  checksum, known-good result). A run that merely exits 0 proves nothing.
