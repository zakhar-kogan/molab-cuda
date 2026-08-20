"""Provision a usable CUDA toolkit inside a Molab (or similar) sandbox.

Molab GPU images ship the NVIDIA driver and PyTorch wheels but no compiler
toolkit: no ``nvcc``, ``cmake``, ``ninja`` or CUDA headers. That is fine for the
overwhelming majority of notebooks, which call precompiled kernels through
PyTorch, but it blocks anything that compiles CUDA from source at runtime.

This module builds a version-matched toolkit from NVIDIA's pip wheels into a
private prefix, repairs the differences between a wheel layout and a real
toolkit, and hands back the environment overrides a build needs.

Typical use::

    toolkit = ensure_toolkit()          # idempotent; skips when already complete
    verify(toolkit)                     # compiles AND runs a kernel for this GPU
    subprocess.run(cmd, env={**os.environ, **toolkit.env})

Why a private prefix: PyTorch pins its own CUDA runtime wheels. Upgrading those
in place to satisfy a compiler is a good way to break torch. Everything here
installs to ``prefix`` and touches nothing else.

Status: extracted from a working Molab port but NOT yet re-run end to end as a
module. Treat the first execution as the real test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

__all__ = ['Toolkit', 'detect_arch', 'ensure_toolkit', 'verify', 'bundle', 'restore']

# One minor version for every compiler-side component. Mixing minors is the
# subtle failure: nvcc 13.3 against 13.0 runtime headers trips CCCL's
# compiler/headers compatibility assert, which reports itself as an include-path
# problem and sends you looking in the wrong place.
DEFAULT_VERSION = '13.3.*'

# The `nvidia-*-cu13` spellings are deprecated and fail to build; these bare
# names are the current ones.
TOOLKIT_PACKAGES = (
    'nvidia-cuda-nvcc', 'nvidia-cuda-runtime', 'nvidia-cuda-crt',
    'nvidia-nvvm', 'nvidia-cuda-cccl', 'nvidia-cuda-nvrtc',
    'nvidia-nvjitlink', 'nvidia-nvtx',
)

# Math libraries version independently of the toolkit minor, so they are pinned
# separately (unpinned). cuRAND is pulled in by CUTLASS's fused-attention
# example; cuBLAS by most LibTorch-adjacent code.
LIBRARY_PACKAGES = ('nvidia-curand', 'nvidia-cublas')

BUILD_TOOLS = ('cmake', 'ninja')

# Headers whose absence only surfaces deep into a build. Checking `nvcc` alone
# is not enough - that check passes while the build still dies on NVTX or
# cuRAND several minutes in.
REQUIRED_FILES = (
    'bin/nvcc',
    'nvvm/bin/cicc',
    'include/cuda_runtime.h',
    'include/cccl/cub/cub.cuh',
    'include/nvtx3/nvToolsExt.h',
    'include/curand_kernel.h',
    'include/cublas_v2.h',
    # Libraries, not just headers. A tree can carry every header and still fail
    # to link minutes into a build: observed on Molab, where persistent storage
    # dropped libnvrtc.so.13 and nvvm/bin/cicc while leaving the rest intact.
    'lib/libnvrtc.so',
    'lib/libcudart.so',
    'lib/libnvJitLink.so',
)

DRIVER_DIRS = ('/usr/lib/x86_64-linux-gnu', '/usr/lib64')


@dataclass
class Toolkit:
    root: Path
    arch: str
    env: dict[str, str] = field(default_factory=dict)

    @property
    def nvcc(self) -> Path:
        return self.root / 'bin' / 'nvcc'


def detect_arch() -> str:
    """Compute capability of the attached GPU, e.g. '120' for sm_120.

    Always detect rather than assume: projects commonly default to sm_75/86,
    which silently produces a binary that will not run on a newer card.
    """
    query = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True)
    caps = {line.strip().replace('.', '') for line in query.splitlines() if line.strip()}
    if not caps:
        raise RuntimeError('no GPU detected; attach a GPU compute spec')
    if len(caps) != 1:
        raise RuntimeError(f'heterogeneous GPU architectures are unsupported: {sorted(caps)}')
    arch = caps.pop()
    if not arch.isdigit():
        raise RuntimeError(f'cannot parse compute capability: {arch!r}')
    return arch


def _pip_install(packages, target: Path | None, cache_dir: Path,
                 force: bool = False) -> None:
    command = [sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
               '--cache-dir', str(cache_dir)]
    if force:
        command.append('--force-reinstall')
    if target is not None:
        command += ['--target', str(target)]
    subprocess.run([*command, *packages], check=True)


def _missing(root: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def _repair_layout(root: Path) -> list[str]:
    """Make a wheel tree behave like a toolkit tree.

    Wheels ship only versioned sonames (``libcudart.so.13``), so the linker
    cannot resolve ``-lcudart``. The driver library lives outside the prefix
    entirely, and CMake's FindCUDAToolkit still looks for ``lib64``.
    """
    repaired: list[str] = []
    lib = root / 'lib'
    if lib.is_dir():
        for shared_object in sorted(lib.glob('*.so.*')):
            alias = lib / (shared_object.name.split('.so')[0] + '.so')
            if not alias.exists():
                alias.symlink_to(shared_object.name)
                repaired.append(alias.name)
        if not (lib / 'libcuda.so').exists():
            driver = [candidate
                      for directory in DRIVER_DIRS
                      for candidate in sorted(Path(directory).glob('libcuda.so*'))
                      if Path(directory).is_dir()]
            if not driver:
                raise RuntimeError('NVIDIA driver library not found; is a GPU attached?')
            (lib / 'libcuda.so').symlink_to(driver[0])
            repaired.append('libcuda.so')
    if not (root / 'lib64').exists():
        (root / 'lib64').symlink_to('lib')
        repaired.append('lib64')
    return repaired


def ensure_toolkit(prefix: str | os.PathLike = '/marimo/storage/cuda-toolkit',
                   version: str = DEFAULT_VERSION,
                   cache_dir: str | os.PathLike | None = None,
                   arch: str | None = None) -> Toolkit:
    """Install (or reuse) a complete CUDA toolkit and return it.

    ``prefix`` should live on whatever storage outlives the kernel. ``cache_dir``
    defaults to a sibling of the prefix: the sandbox's default pip cache is
    typically unwritable, which disables caching silently and re-downloads
    gigabytes of wheels on every cold start.
    """
    prefix = Path(prefix)
    cache_dir = Path(cache_dir) if cache_dir else prefix.parent / 'pip-cache'
    root = prefix / 'nvidia' / 'cu13'
    for directory in (prefix, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Repair first: the layout check below counts symlinks, and on lossy storage
    # a dropped symlink is a repair, not a reinstall.
    if root.is_dir():
        _repair_layout(root)

    if _missing(root):
        # --force-reinstall because pip reads dist-info, not payload. When
        # storage drops bin/ but leaves nvidia_cuda_nvcc-*.dist-info behind,
        # a plain install reports success and changes nothing.
        pinned = [f'{name}=={version}' for name in TOOLKIT_PACKAGES]
        _pip_install([*pinned, *LIBRARY_PACKAGES], prefix, cache_dir,
                     force=True)
        if root.is_dir():
            _repair_layout(root)
    still_missing = _missing(root)
    if still_missing:
        raise RuntimeError(f'CUDA toolkit install is incomplete: {still_missing}')

    absent = [name for name in BUILD_TOOLS if shutil.which(name) is None]
    if absent:
        _pip_install(absent, None, cache_dir)

    arch = arch or detect_arch()
    env = {
        'CUDA_HOME': str(root),
        'CUDA_PATH': str(root),
        'CUDACXX': str(root / 'bin' / 'nvcc'),
        'PATH': f"{root / 'bin'}:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
        'NVCC_PREPEND_FLAGS': f"-I{root / 'include' / 'cccl'}",
        'LD_LIBRARY_PATH': f"{root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(':'),
        'PIP_CACHE_DIR': str(cache_dir),
    }
    return Toolkit(root=root, arch=arch, env=env)


def verify(toolkit: Toolkit, workdir: str | os.PathLike | None = None) -> None:
    """Compile and run a kernel for the detected architecture.

    Running it matters: a version-skewed toolkit can fail at the first CCCL
    include, and a mis-detected architecture produces a binary that builds
    cleanly and then refuses to launch. Only an executed kernel proves both.
    """
    workdir = Path(workdir or os.environ.get('TMPDIR', '/tmp')) / 'molab_cuda_check'
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / 'check.cu'
    source.write_text(
        '#include <cub/cub.cuh>\n'
        '#include <nvtx3/nvToolsExt.h>\n'
        '__global__ void probe(int* out) { *out = threadIdx.x; }\n'
        'int main() {\n'
        '  int* device = nullptr;\n'
        '  if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 1;\n'
        '  probe<<<1, 1>>>(device);\n'
        '  return cudaDeviceSynchronize() == cudaSuccess ? 0 : 2;\n'
        '}\n', encoding='utf-8')
    binary = workdir / 'check'
    environment = {**os.environ, **toolkit.env}
    compile_result = subprocess.run(
        [str(toolkit.nvcc), f'-arch=sm_{toolkit.arch}', '-std=c++17', str(source),
         '-o', str(binary), '-lcudart', '-L', str(toolkit.root / 'lib')],
        capture_output=True, text=True, env=environment)
    if compile_result.returncode != 0:
        raise RuntimeError(f'toolkit failed to compile for sm_{toolkit.arch}:\n'
                           f'{compile_result.stderr[-2000:]}')
    run_result = subprocess.run([str(binary)], capture_output=True, text=True, env=environment)
    if run_result.returncode != 0:
        raise RuntimeError(
            f'kernel compiled for sm_{toolkit.arch} but failed to run '
            f'(exit {run_result.returncode}); architecture or driver mismatch')


def bundle(toolkit: Toolkit, archive: str | os.PathLike) -> Path:
    """Pack the toolkit for restoring into a fresh sandbox.

    Sandboxes do not share storage, so a cold start otherwise repeats the whole
    download. Symlinks are stored as links, not followed, so the archive stays
    close to the installed size; ``restore`` re-runs the layout repair anyway
    because ``libcuda.so`` must point at the new sandbox's driver.
    """
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(toolkit.root, arcname='cu13')
    return archive


def restore(archive: str | os.PathLike,
            prefix: str | os.PathLike = '/marimo/storage/cuda-toolkit',
            arch: str | None = None) -> Toolkit:
    """Unpack a bundle produced by :func:`bundle` and return a usable toolkit."""
    prefix = Path(prefix)
    root = prefix / 'nvidia' / 'cu13'
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(root.parent)
    missing = _missing(root)
    if missing:
        raise RuntimeError(f'restored toolkit is incomplete: {missing}')
    _repair_layout(root)
    return ensure_toolkit(prefix=prefix, arch=arch)
