# Running CayleyPy on Molab

Point an agent here: *"Use https://github.com/zakhar-kogan/molab-cuda/blob/master/cayleypy.md."*

```
Run the CayleyPy beam-search solver on molab: <MOLAB_URL> token <TOKEN>
Connect with /marimo-pair.

- Toolkit: github.com/zakhar-kogan/molab-cuda — ensure_toolkit() + verify().
- Solver: zakhar-kogan/MultiGPUBeamSearch @ fix/single-gpu-and-build-portability
  (upstream crashes on 1 GPU, builds with -j2, reuses stale binaries).
- Competition test.csv needs Kaggle creds (molab Secrets panel); the checkpoint
  downloads token-free. puzzle_info.json can be rebuilt from the notebook's own
  symmetry tables.
- Cache under /marimo/storage/cayleypy/: toolkit, PIP_CACHE_DIR, kagglehub, repo,
  build. Key the build dir by solver commit + arch + backend.
- Read the molab facts first: PORTING.md.

Done: a solve whose path passes exact replay.
```

Beam sizing is per-card and the shipped runtime profiles are `kaggle_2xt4`
anchors — treat them as a starting point, not a tuned configuration.

See [PORTING.md](PORTING.md) for the molab facts and for porting any other
Kaggle GPU notebook.
