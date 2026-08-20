# molab-cuda

Get a working `nvcc` on a GPU notebook sandbox that has the NVIDIA driver but no
CUDA toolkit.

```python
import molab_cuda

toolkit = molab_cuda.ensure_toolkit()   # idempotent
molab_cuda.verify(toolkit)              # compiles AND runs a kernel for this GPU

subprocess.run(build_command, env={**os.environ, **toolkit.env})
```

Copy-pasteable agent prompts: [cayleypy.md](cayleypy.md) to run the CayleyPy
solver, [PORTING.md](PORTING.md) to port any Kaggle GPU notebook — both with the
Molab facts worth knowing before you start.

## Why this exists

[Molab](https://molab.marimo.io) GPU sandboxes ship the driver and PyTorch
wheels, but no compiler toolkit: no `nvcc`, `cmake`, `ninja` or CUDA headers.
That is a reasonable image choice — almost every notebook calls precompiled
kernels through PyTorch, and a full toolkit is several GB nobody would use.
Colab is much the same. Kaggle happens to include one.

It only becomes a problem if you compile CUDA from source at runtime, at which
point you get:

```
FileNotFoundError: [Errno 2] No such file or directory: 'cmake'
```

Assembling a toolkit from NVIDIA's pip wheels works, but several details are
not obvious and each one costs a debugging round. This module encodes them.

## What it handles

**Package naming.** The `nvidia-*-cu13` distributions are deprecated and fail to
build. The current names are unsuffixed (`nvidia-cuda-nvcc`, …).

**Version skew.** The CUDA wheels PyTorch pulls in are not a usable toolkit:
`nvcc`, the runtime headers and CCCL arrive at mismatched minor versions, and
CCCL's compatibility check then fails with

```
error: "CUDA compiler and CUDA toolkit headers are incompatible,
        please check your include paths"
```

which sends you hunting through include paths when the real problem is that
nvcc is 13.3 and the runtime headers are 13.0. Everything compiler-side is
pinned to a single minor here.

**Not disturbing PyTorch.** Torch pins its own CUDA runtime wheels; upgrading
them in place to satisfy a compiler is a good way to break torch. Everything
installs into a private prefix.

**Missing sonames.** Wheels ship only versioned libraries (`libcudart.so.13`),
so `-lcudart` fails to link. The unversioned symlinks are created, along with
`libcuda.so` pointing at the driver (which lives outside the prefix) and a
`lib64` alias for CMake's `FindCUDAToolkit`.

**Headers that surface late.** A completeness check on `nvcc` alone passes while
the build still dies minutes later on a missing `nvtx3/nvToolsExt.h` or
`curand_kernel.h`. Every required header is checked up front.

**Architecture detection.** Read from `nvidia-smi` rather than assumed. Projects
commonly default to `sm_75`/`sm_86`; a current card may be `sm_120`, and the
mismatch produces a binary that builds cleanly and refuses to launch.

**pip's cache.** Molab leaves `~/.cache/pip` unwritable, which silently disables
caching and re-downloads gigabytes on every cold start. The cache is redirected
to a writable path.

## Persistence, and a warning about it

Point `prefix` at storage that outlives the kernel (`/marimo/storage/...` on
Molab) and a restart reuses the toolkit instead of refetching ~1 GB.

**Molab's persistent storage is lossy.** It survives a sandbox, but not
faithfully. Observed across a single sandbox hop: symlinks dropped throughout a
939 MB toolkit; a CUTLASS checkout that kept `.git` and `examples/` but lost
`include/`; a repo that kept `tools/` but lost `.git`, `configs/`, `cuda/` and
`src/`; a `kagglehub` directory that kept its structure but lost a 12 MB
`model.pth`.

Each produced a different and confusing failure, all with the same root cause:
**a half-present cache is worse than an absent one**, because tools check for
existence and skip the repair. Anything caching on this storage should validate
by content, not by presence.

`ensure_toolkit()` re-runs the layout repair every call for exactly this reason,
so a dropped symlink is fixed rather than inherited.

## Moving between sandboxes

```python
molab_cuda.bundle(toolkit, "/somewhere/cuda-13.3.tar.gz")
toolkit = molab_cuda.restore("/somewhere/cuda-13.3.tar.gz")
```

`restore()` deliberately re-runs the layout repair rather than trusting the
archive: `libcuda.so` points at the driver, which differs per sandbox, so an
unpacked bundle would otherwise carry a dangling symlink onto a machine where it
is wrong.

## Status and limits

Extracted from a working port of a CUDA/C++ beam-search solver onto Molab, where
it builds and runs for `sm_120`.

- `ensure_toolkit()` and `verify()` are exercised: they have provisioned a
  toolkit from scratch, reused a persisted one, repaired symlinks a sandbox hop
  had dropped, and compiled and run an `sm_120` kernel.
- `bundle()` and `restore()` have **never been executed end to end**. They were
  written against the persistence problem described above, and the first run
  will be their real test. Treat them as a sketch.
- Linux x86_64 only. Assumes the CUDA 13 wheel layout (`nvidia/cu13/...`).
- Assumes `nvidia-smi` and a driver new enough for the toolkit minor.
- Not affiliated with NVIDIA or marimo.

Issues and patches welcome — particularly reports from other images, drivers and
CUDA versions, which is where this is most likely to be wrong.

## Licence

MIT
