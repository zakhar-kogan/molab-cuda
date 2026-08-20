# Running CUDA work on Molab

Point an agent here: *"Use https://github.com/zakhar-kogan/molab-cuda/blob/master/PORTING.md."*

## Run CayleyPy on molab

```
Run the CayleyPy beam-search solver on molab: <MOLAB_URL> token <TOKEN>
Connect with /marimo-pair.

- Toolkit: github.com/zakhar-kogan/molab-cuda — ensure_toolkit() + verify().
- Solver: zakhar-kogan/MultiGPUBeamSearch @ fix/single-gpu-and-build-portability
  (upstream crashes on 1 GPU, builds with -j2, reuses stale binaries).
- Competition test.csv needs Kaggle creds (molab Secrets panel); the checkpoint
  downloads token-free. puzzle_info.json can be rebuilt from the notebook's own
  symmetry tables.
- Read the molab facts below before starting.

Done: a solve whose path passes exact replay.
```

## Port a Kaggle notebook to molab

```
Port this Kaggle GPU notebook to molab: <MOLAB_URL> token <TOKEN>
Connect with /marimo-pair. Source: <notebook/repo>

- Toolkit: github.com/zakhar-kogan/molab-cuda — ensure_toolkit() + verify().
- Never edit the .py; drive the live kernel via marimo._code_mode.
- Detect GPU count and arch. Kaggle code assumes 2 GPUs and sm_75/86 — expect the
  solver to finish correctly and then crash in reporting.
- Split assets into credential-gated vs synthesizable from the notebook itself.
- Cache under /marimo/storage/<project>/: toolkit, PIP_CACHE_DIR, kagglehub, repo,
  build. Key build dirs by commit + arch + backend.
- Read the molab facts below before starting.

Done: runs end-to-end verified, warm restart with zero compiles/downloads.
```

## Molab facts

- **No toolkit.** Driver and torch are present; `nvcc`, `cmake`, `ninja` are not.
  Surfaces as a bare `FileNotFoundError: 'cmake'`. That's what this repo fixes.
- **Storage is lossy.** `/marimo/storage` survives a sandbox but silently drops
  symlinks, `.git` and large files — so tools see a populated cache and skip the
  repair, failing minutes later somewhere unrelated. Validate caches by content,
  delete partials. A half-present cache is worse than none.
- **pip's cache is off.** `~/.cache/pip` is unwritable, so gigabytes re-download
  every cold start unless you set `PIP_CACHE_DIR`.
- **Cells die with the connection.** Marimo interrupts a running cell when the
  client disconnects — run long builds and solves detached (`nohup`, new session).
- **Sandboxes die.** Three terminated mid-work while this was written. Assume any
  long job can vanish; make restarts cheap.
- **Exit 0 proves nothing.** Gate on the work's own correctness check — exact
  replay, checksum, known-good result.
