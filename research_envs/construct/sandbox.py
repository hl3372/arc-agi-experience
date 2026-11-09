"""Run a model-written program (a witness construction or a solver's attempt) under a kernel-enforced sandbox.

Isolation uses Landlock (unprivileged, enforced by the kernel; no user namespaces needed, so it works where an
AppArmor policy blocks bubblewrap). Before exec, the child process:
- sets no_new_privs and restricts itself so it can only read /usr, /etc, /lib, /lib64, the Python installation and
  the directories holding numpy, networkx and ortools, and read/write its own fresh working directory (`sbx_*` under
  the temporary directory, holding `spec.json` and `main.py`). Home, this project, runs/, the archive, other
  candidates and private files are unreadable;
- denies all TCP bind and connect, and scopes signals and abstract unix sockets to itself. Landlock has no rules
  for other protocols: UDP datagrams (DNS included) still leave the sandbox, so this is not full network isolation
  (run rounds without network access if that matters);
- runs in its own session, with memory and CPU limits and a wall-clock timeout.

Requirements: Linux with Landlock ABI 6 or later (kernel 6.12+, Landlock enabled in the LSM list). Anywhere else the
sandbox refuses to run programs (SandboxUnavailable), so domain B cannot build witnesses or score code pilots: it fails
closed, never open.

The program prints the candidate object as JSON on its last stdout line. The output is then checked by the trusted
evaluator like any other candidate; the program itself is never trusted.
"""
import ctypes
import importlib.util
import json
import os
import resource
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

MIN_ABI = 6
SANDBOX_PACKAGES = ('numpy', 'networkx', 'ortools')    # what witness and pilot programs are told they can import


class SandboxUnavailable(RuntimeError):
    pass


def _python():
    """The base interpreter, outside any virtual environment, so the child needs no read access to the venv itself."""
    return os.path.realpath(getattr(sys, '_base_executable', None) or sys.executable)


PYTHON = _python()

_SYS_CREATE, _SYS_ADD, _SYS_RESTRICT = 444, 445, 446
_PR_SET_NO_NEW_PRIVS = 38
_FS_ALL = (1 << 16) - 1                      # every filesystem right up to IOCTL_DEV (ABI >= 5)
_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1, 2, 4, 8
_TRUNCATE, _IOCTL_DEV = 1 << 14, 1 << 15
_RO = _EXECUTE | _READ_FILE | _READ_DIR
_NET_ALL = 1 | 2                              # BIND_TCP | CONNECT_TCP (ABI >= 4)
_SCOPE_ALL = 1 | 2                            # abstract unix sockets | signals (ABI >= 6)


def landlock_abi():
    """The kernel's Landlock ABI version, or -1 where there is none (another OS, or Landlock disabled)."""
    if not sys.platform.startswith('linux'):
        return -1
    libc = ctypes.CDLL(None, use_errno=True)
    return libc.syscall(_SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))


def check_available():
    abi = landlock_abi()
    if abi < MIN_ABI:
        raise SandboxUnavailable(
            f'the program sandbox needs Linux with Landlock ABI >= {MIN_ABI} (kernel 6.12+); this system reports '
            f'{abi if abi >= 0 else "no Landlock"}. Model-written programs are never run unsandboxed.')
    return abi


def _restrict(read_only, read_write, devices):
    """Called in the child before exec. Raises on any failure, so the program never runs unrestricted."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'no_new_privs')
    attr = struct.pack('=QQQ', _FS_ALL, _NET_ALL, _SCOPE_ALL)
    buf = ctypes.create_string_buffer(attr, len(attr))
    fd = libc.syscall(_SYS_CREATE, buf, ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')

    def allow(path, rights):
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = struct.pack('=Qi', rights, pfd)
            rb = ctypes.create_string_buffer(rule, len(rule))
            if libc.syscall(_SYS_ADD, ctypes.c_int(fd), ctypes.c_int(1), rb, ctypes.c_uint32(0)) != 0:
                raise OSError(ctypes.get_errno(), f'landlock_add_rule {path}')
        finally:
            os.close(pfd)

    for p in read_only:
        if os.path.exists(p):
            allow(p, _RO)
    for p in read_write:
        allow(p, _FS_ALL)
    for p in devices:
        allow(p, _READ_FILE | _WRITE_FILE | _TRUNCATE | _IOCTL_DEV)
    if libc.syscall(_SYS_RESTRICT, ctypes.c_int(fd), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    os.close(fd)


def site_dirs():
    """Directories the child may import from: wherever numpy, networkx and ortools are installed, plus this
    environment's site-packages. Their .pth files are not processed in the child (plain sys.path entries)."""
    dirs = []
    for name in SANDBOX_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            dirs.append(str(Path(list(spec.submodule_search_locations)[0]).resolve().parent))
    for key in ('purelib', 'platlib'):
        p = sysconfig.get_paths().get(key)
        if p and os.path.isdir(p):
            dirs.append(str(Path(p).resolve()))
    return list(dict.fromkeys(dirs))


def _protected():
    """Folders that must stay unreadable: a read rule on any of them or their ancestors would expose them."""
    from ..common import ROOT, CHECKOUT
    out = [Path.home().resolve(), (ROOT / 'runs').resolve()]
    if CHECKOUT:
        out.append(CHECKOUT.resolve())
    return out


def read_only_paths():
    paths = ['/usr', '/etc', '/lib', '/lib64', '/proc/self']
    prefix = os.path.realpath(sys.base_prefix)
    if not any(prefix == p or prefix.startswith(p + '/') for p in ('/usr', '/lib', '/lib64')):
        paths.append(prefix)     # a Python installed outside /usr (pyenv, conda, uv)
    paths += site_dirs()
    for p in paths:
        rp = Path(os.path.realpath(p))
        for q in _protected():
            if rp == q or rp in q.parents:
                raise SandboxUnavailable(f'refusing a sandbox read rule on {rp}: it would expose {q}. Install Python '
                                         f'and the packages outside your home folder and this project, or use a '
                                         f'virtual environment.')
    return list(dict.fromkeys(paths))


def _child_setup(work, mem_mb, cpu_s):
    read_only = read_only_paths()

    def setup():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        _restrict(read_only=read_only, read_write=[str(work)], devices=['/dev/null', '/dev/urandom'])
    return setup


def run_program(code, public_spec, timeout=120, mem_mb=2048, keep=None):
    """Run `code` (Python source) against the public spec. Returns {"ok", "object", "stdout_tail", "stderr_tail",
    "returncode", "timed_out"}. `keep` copies the working directory there afterwards (for dossiers)."""
    check_available()
    sites = site_dirs()
    work = Path(tempfile.mkdtemp(prefix='sbx_'))
    try:
        (work / 'main.py').write_text(code)
        (work / 'spec.json').write_text(json.dumps(public_spec))
        setup = _child_setup(work, mem_mb, timeout + 5)
        env = {'HOME': str(work), 'PYTHONHASHSEED': '0', 'PATH': '/usr/bin',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        timed_out = False
        try:
            # the package dirs go after the standard library and before any system site-packages
            boot = (f'import sys; i = next((k for k, p in enumerate(sys.path) if p.endswith(("site-packages", '
                    f'"dist-packages"))), len(sys.path)); sys.path[i:i] = {sites!r}; '
                    'exec(compile(open("main.py").read(), "main.py", "exec"))')
            p = subprocess.run([PYTHON, '-I', '-c', boot],
                               cwd=work, env=env, capture_output=True, text=True, timeout=timeout, preexec_fn=setup)
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            timed_out, rc = True, None
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or '')
        obj, ok = None, False
        if not timed_out:
            lines = [l for l in stdout.strip().splitlines() if l.strip()]
            if lines:
                try:
                    obj, ok = json.loads(lines[-1]), True
                except json.JSONDecodeError:
                    pass
        if keep:
            shutil.copytree(work, keep, dirs_exist_ok=True)
        return {'ok': ok, 'object': obj, 'stdout_tail': stdout[-2000:], 'stderr_tail': stderr[-2000:],
                'returncode': rc, 'timed_out': timed_out}
    finally:
        shutil.rmtree(work, ignore_errors=True)
