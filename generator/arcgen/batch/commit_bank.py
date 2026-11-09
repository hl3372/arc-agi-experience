"""Bank-scale helpers of the v1 commit (speed-up of 2026-10-24).

Each helper keeps the result of the code it replaces and only removes repeated bank-wide work:

- load_registry: the canonical loader (committed_tasks.load_committed_task_registry), with dossier verification either
  deep (every artifact of every dossier hashed, prefetched by threads in registry order) or shallow (the dossier
  manifest's own hash, binding and file list). A commit runs the deep form once, after the install, on the final bank.
- the render index (committed_tasks/RENDER_SHA256SUMS, `sha256sum -c` format): sha256 of every file under renders/,
  so a commit draws only its new tasks and checks the renders it keeps by hash instead of drawing the bank again.
- plan_renders: the canonical _render_bank (authorized_commits) run with its drawing deferred: it writes the render
  manifest and README exactly as before and returns what it would draw and put on the contact sheet.
- contact_sheet: commit.bounded_contact_sheet, byte for byte, with the thumbnails made in worker processes.
- link_or_copy: model-call and terminal-evidence records hard-linked (and made read-only) instead of copied.
- backup / rollback: a backup of exactly the bank files an install replaces or removes.
"""
from __future__ import annotations

import errno
import gc
import math
import os
import shutil
import stat
import struct
import threading
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

from arcgen.lib import authorized_commits, committed_dossiers
from arcgen.lib.committed_dossiers import CommittedDossierError, DOSSIER_SCHEMA_VERSION
from arcgen.lib.committed_tasks import load_committed_task_registry
from arcgen.lib.util import atomic_write_json, read_json

RENDER_INDEX = 'RENDER_SHA256SUMS'
SHARED_RENDER_FILES = ('manifest.json', 'README.md', 'contact_sheet.png')
THUMBNAIL = (760, 720)
# Files a run may still rewrite; never hard-linked into a dossier.
MUTABLE_NAMES = ('STATUS.json', 'runner.lock')
MUTABLE_SUFFIXES = ('.sqlite3', '-wal', '-shm', '.lock')


def workers():
    """Worker processes for drawing renders and thumbnails (ARCGEN_COMMIT_WORKERS; default a third of the CPUs, at most 6)."""
    value = os.environ.get('ARCGEN_COMMIT_WORKERS')
    return max(1, int(value)) if value else max(1, min(6, (os.cpu_count() or 2) // 3))


def io_threads():
    """Threads hashing files (ARCGEN_COMMIT_IO_THREADS; default 8)."""
    return max(1, int(os.environ.get('ARCGEN_COMMIT_IO_THREADS') or 8))


def verify_processes():
    """Processes verifying dossiers (ARCGEN_COMMIT_VERIFY_PROCESSES; default 8). The check is bound by Python work per file
    (path resolution, JSON), not by the disk: 8 processes read the real bank at about 1.3 GB/s, 8 threads at 0.3 GB/s."""
    return max(1, int(os.environ.get('ARCGEN_COMMIT_VERIFY_PROCESSES') or 8))


def file_sha256(path):
    digest = sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths, threads=None):
    """{path: sha256} for many files, read by a thread pool."""
    paths = list(paths)
    if len(paths) < 64:
        return {p: file_sha256(p) for p in paths}
    with ThreadPoolExecutor(threads or io_threads()) as pool:
        return dict(zip(paths, pool.map(file_sha256, paths)))


def _sha_or_error(path):
    try:
        return file_sha256(path)
    except Exception as exc:
        return exc


def hash_each(paths, threads=None):
    """[sha256 or the exception reading it] per path, in order (a thread pool reads ahead), so a caller checking them
    in order fails on the same first file as reading them one by one."""
    paths = list(paths)
    if len(paths) < 64:
        return [_sha_or_error(p) for p in paths]
    with ThreadPoolExecutor(threads or io_threads()) as pool:
        return list(pool.map(_sha_or_error, paths))


@contextmanager
def patched(module, **values):
    """Temporarily replace module attributes (single-threaded callers only)."""
    saved = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


# ------------------------------------------------------------------ registry
def shallow_dossier_check(*, committed_root, entry):
    """committed_dossiers.verify_dossier_manifest without hashing the dossier's artifact files: the manifest's own hash
    (bound in the registry), its identity and visibility, and a well-formed contained file list with the required records."""
    relative, expected_hash = entry.get('dossier_path'), entry.get('dossier_sha256')
    if not isinstance(relative, str) or not relative.strip():
        raise CommittedDossierError('committed entry has no dossier_path')
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise CommittedDossierError('committed entry has no valid dossier_sha256')
    manifest_path = committed_dossiers._contained(committed_root, relative)
    if not manifest_path.is_file():
        raise CommittedDossierError(f'dossier manifest is missing: {manifest_path}')
    actual_hash = committed_dossiers._file_sha256(manifest_path)
    if actual_hash != expected_hash:
        raise CommittedDossierError(f'dossier manifest hash mismatch: {actual_hash} != {expected_hash}')
    manifest = read_json(manifest_path)
    if manifest.get('schema_version') != DOSSIER_SCHEMA_VERSION:
        raise CommittedDossierError('unsupported dossier schema')
    if manifest.get('registry_task_id') != entry.get('registry_task_id'):
        raise CommittedDossierError('dossier is bound to a different registry task')
    if manifest.get('arc_sha256') != entry.get('arc_sha256'):
        raise CommittedDossierError('dossier is bound to a different ARC payload')
    if manifest.get('visibility') != 'author_only' or manifest.get('contains_hidden_test_output') is not True:
        raise CommittedDossierError('dossier visibility declaration is invalid')
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise CommittedDossierError('dossier manifest has no artifact files')
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise CommittedDossierError('dossier file entry is malformed')
        path_text = item.get('path')
        if not isinstance(path_text, str) or path_text in seen:
            raise CommittedDossierError('dossier file path is missing or duplicated')
        seen.add(path_text)
        # lexical containment (no file is opened here); the deep check resolves every path, symlinks included
        normal = os.path.normpath(path_text)
        if os.path.isabs(path_text) or normal == '..' or normal.startswith('../'):
            raise CommittedDossierError(f'path escapes repository root: {path_text}')
    required = {'authoring.json', 'review.json', 'provenance.json'}
    if not required.issubset(seen):
        raise CommittedDossierError(f"dossier lacks required artifacts: {', '.join(sorted(required - seen))}")
    return manifest


# the canonical check, captured before any patching: worker processes always run it
_CANONICAL_DOSSIER_CHECK = committed_dossiers.verify_dossier_manifest


def _dossier_outcome(root, entry):
    try:
        return True, _CANONICAL_DOSSIER_CHECK(committed_root=Path(root), entry=entry)
    except Exception as exc:  # handed back to the loader, which raises it in registry order
        return False, exc


class _Prefetch:
    """committed_dossiers.verify_dossier_manifest results computed ahead by worker processes (the unchanged function),
    handed to the loader in registry order through a bounded window. Any call the window cannot answer runs the real
    check in this process."""

    def __init__(self, root, entries, real, processes, window=256):
        self.root, self.real, self.window = Path(root), real, window
        self.entries, self.pending = iter(entries), deque()
        self.workers = _Pool(processes)
        self.pool = self.workers.executor or ThreadPoolExecutor(1)
        self.fill()  # starts the workers before the loader's check is patched

    @staticmethod
    def key(root, entry):
        return (str(Path(root)),) + tuple(repr(entry.get(k)) for k in ('registry_task_id', 'dossier_path', 'dossier_sha256', 'arc_sha256'))

    def fill(self):
        while len(self.pending) < self.window:
            entry = next(self.entries, None)
            if entry is None:
                return
            if isinstance(entry, dict):
                self.pending.append((self.key(self.root, entry), self.pool.submit(_dossier_outcome, str(self.root), entry)))

    def __call__(self, *, committed_root, entry):
        wanted = self.key(committed_root, entry)
        while self.pending:
            key, future = self.pending.popleft()
            self.fill()
            if key == wanted:
                ok, value = future.result()
                if not ok:
                    raise value
                return value
        return self.real(committed_root=committed_root, entry=entry)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.workers.close()


def load_registry(path, *, deep, processes=None):
    """load_committed_task_registry(path) with deep (every dossier artifact hashed by the canonical check, in worker
    processes) or shallow dossier checks. Every other check of the loader (fields, uniqueness, task hashes, ARC validity,
    commit-set counts) runs unchanged, in order."""
    path = Path(path).resolve()
    if not deep:
        with patched(committed_dossiers, verify_dossier_manifest=shallow_dossier_check):
            return load_committed_task_registry(path)
    real = committed_dossiers.verify_dossier_manifest
    entries = (read_json(path) or {}).get('entries') or []
    prefetch = _Prefetch(path.parent, entries, real, processes or verify_processes())
    try:
        with patched(committed_dossiers, verify_dossier_manifest=prefetch):
            return load_committed_task_registry(path)
    finally:
        prefetch.close()


# ------------------------------------------------------------------ render index
def read_render_index(bank):
    """{render file name: sha256} from committed_tasks/RENDER_SHA256SUMS, or None when the bank has none."""
    path = Path(bank) / RENDER_INDEX
    if not path.is_file():
        return None
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        digest, sep, name = line.partition('  ')
        if (not sep or len(digest) != 64 or not name.startswith('renders/') or '/' in name[8:] or name[8:] in rows
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Malformed render index: ' + line[:120])
        rows[name[8:]] = digest
    return rows


def format_render_index(rows):
    return ''.join(f'{rows[name]}  renders/{name}\n' for name in sorted(rows))


def plan_renders(destination, registry):
    """Run authorized_commits._render_bank(destination, registry) with drawing deferred. It writes renders/manifest.json
    and renders/README.md under destination byte for byte as the full render does, reads no task, draws nothing, and
    returns (draws, images): draws = one job per registry entry in order ({'name', 'task_path', 'title', 'subtitle',
    'rule'}); images = [(label, render file name)] in contact-sheet order."""
    draws, images = [], []

    def draw(task, *, title, subtitle, rule=None, destination):
        draws.append({'name': Path(destination).name, 'task_path': str(task), 'title': title, 'subtitle': subtitle, 'rule': rule})
        return Path(destination)

    def sheet(pairs, _destination):
        images.extend((label, Path(path).name) for label, path in pairs)

    with patched(authorized_commits, render_episode=draw, build_contact_sheet=sheet, read_json=lambda p: Path(p)):
        authorized_commits._render_bank(Path(destination), registry)
    return draws, images


def _draw(job):
    from arcgen.lib.lean_v3_rendering import render_episode
    render_episode(read_json(Path(job['task_path'])), title=job['title'], subtitle=job['subtitle'], rule=job['rule'],
                   destination=Path(job['out']))
    return job['out']


class _Pool:
    """Forked worker processes, or None when that is not safe (other threads alive: fork could deadlock), in which case the
    caller works in this process. Fork, not spawn: a spawned worker re-runs the parent's __main__ file, and a script that
    commits at top level would commit again in every worker. The parent's objects are frozen out of the garbage collector
    while workers live (gc.freeze), so a worker's collector never touches, and so never copies, the inherited heap (the
    104 MB registry is several GB of objects)."""

    def __init__(self, processes):
        self.executor = None
        if processes > 1 and threading.active_count() == 1:
            gc.collect(); gc.freeze()
            self.executor = ProcessPoolExecutor(processes, mp_context=get_context('fork'))

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True); self.executor = None
            gc.unfreeze()

    def __enter__(self):
        return self.executor

    def __exit__(self, *exc):
        self.close()


def draw_renders(jobs, folder, *, processes=None):
    """Draw render jobs (from plan_renders) into folder with render_episode, in worker processes; same bytes as drawing
    them one by one (render_episode is deterministic)."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    jobs = [{**job, 'out': str(folder / job['name'])} for job in jobs]
    with _Pool(processes or workers() if len(jobs) >= 8 else 1) as pool:
        for _ in (pool.map(_draw, jobs, chunksize=4) if pool else map(_draw, jobs)):
            pass
    return [Path(j['out']) for j in jobs]


# ------------------------------------------------------------------ contact sheet
def thumbnail_size(width, height, box=THUMBNAIL):
    """The size Image.thumbnail(box) gives a width x height image (Pillow's preserve_aspect_ratio); the workers check it."""
    x, y = box
    if x >= width and y >= height:
        return width, height

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)
    aspect = width / height
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    return x, y


def _thumbnail(path):
    from PIL import Image
    with Image.open(path) as source:
        thumb = source.convert('RGB'); thumb.thumbnail(THUMBNAIL, Image.Resampling.LANCZOS)
        value = (thumb.size, thumb.tobytes()); thumb.close()
    return value


class _SizeMismatch(Exception):
    pass


def contact_sheet(images, destination, *, processes=None, in_flight=96):
    """Exactly commit.bounded_contact_sheet(images, destination): the same chunks, compressor calls and bytes. Sizes come
    from the PNG headers and thumbnails from worker processes (at most `in_flight` held at once); if a worker's thumbnail
    is not the predicted size the sheet is made the original sequential way instead."""
    from arcgen.batch.commit import bounded_contact_sheet
    with _Pool(processes or workers() if len(images) >= 16 else 1) as pool:
        if pool is None:
            return bounded_contact_sheet(images, destination)
        try:
            return _parallel_contact_sheet(images, destination, pool, in_flight)
        except _SizeMismatch:
            return bounded_contact_sheet(images, destination)


def _parallel_contact_sheet(images, destination, pool, in_flight):
    from PIL import Image, ImageDraw
    from arcgen.lib.lean_v3_rendering import _font
    columns = 2 if len(images) > 1 else 1
    columns = min(columns, len(images))
    sizes = []
    for _, path in images:
        with Image.open(path) as source:
            sizes.append(thumbnail_size(*source.size))
    tile_width = max(w for w, _ in sizes)
    row_heights = [max(h for _, h in sizes[i:i + columns]) for i in range(0, len(sizes), columns)]
    width = 48 + columns * tile_width + (columns - 1) * 24
    height = 48 + sum(row_heights) + len(row_heights) * 30 + (len(row_heights) - 1) * 24
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    compressor = zlib.compressobj()
    with destination.open('wb') as handle:
        queue, upcoming = deque(), iter(range(len(images)))

        def top_up():
            while len(queue) < in_flight:
                index = next(upcoming, None)
                if index is None:
                    return
                queue.append(pool.submit(_thumbnail, str(images[index][1])))

        def chunk(kind, payload):
            handle.write(struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff))

        def band(image):
            pixels = image.tobytes(); stride = width * 3
            for offset in range(0, len(pixels), stride):
                compressed = compressor.compress(b'\x00' + pixels[offset:offset + stride])
                if compressed:
                    chunk(b'IDAT', compressed)
            image.close()
        top_up()
        handle.write(b'\x89PNG\r\n\x1a\n')
        chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
        band(Image.new('RGB', (width, 24), '#EFEFED'))
        for row, row_height in enumerate(row_heights):
            strip = Image.new('RGB', (width, 30 + row_height + 24), '#EFEFED'); draw = ImageDraw.Draw(strip)
            for column in range(columns):
                index = row * columns + column
                if index >= len(images):
                    break
                label = images[index][0]; x = 24 + column * (tile_width + 24)
                draw.text((x, 0), label, fill='#111111', font=_font(17, bold=True))
                size, data = queue.popleft().result(); top_up()
                if tuple(size) != tuple(sizes[index]):
                    raise _SizeMismatch(images[index][1])
                thumb = Image.frombytes('RGB', size, data)
                strip.paste(thumb, (x, 30)); thumb.close()
            band(strip)
        chunk(b'IDAT', compressor.flush()); chunk(b'IEND', b'')
    return destination


# ------------------------------------------------------------------ dossier records
def linkable(source, runs_root):
    """A run's immutable record: inside runs/, not a file a run rewrites (status, ledgers, locks)."""
    source = Path(source)
    try:
        source.relative_to(runs_root)
    except ValueError:
        return False
    return source.name not in MUTABLE_NAMES and not source.name.endswith(MUTABLE_SUFFIXES)


def link_or_copy(source, target, *, link, stats):
    """Hard-link source at target (replacing a file already there) and make the shared inode read-only; fall back to a
    copy (shutil.copy2, as before) when linking is not possible (another filesystem, link limit, permissions). Returns
    True when linked."""
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    if link and target.exists() and os.path.samefile(source, target):
        return True  # already this record (rename() between two links of one inode would do nothing)
    if link:
        temporary = target.with_name('.' + target.name + '.linking')
        try:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            os.link(source, temporary)
        except OSError as exc:
            key = 'link_fallback_' + (errno.errorcode.get(exc.errno, str(exc.errno)) if exc.errno else 'error')
            stats[key] = stats.get(key, 0) + 1
        else:
            os.replace(temporary, target)
            mode = stat.S_IMODE(os.stat(target).st_mode)
            if mode & 0o222:
                try:
                    os.chmod(target, mode & ~0o222)
                except OSError:
                    stats['chmod_failed'] = stats.get('chmod_failed', 0) + 1
            stats['linked'] = stats.get('linked', 0) + 1
            return True
    # a copy under a temporary name, renamed over the target: a file already at target may be a link of source itself
    temporary = target.with_name('.' + target.name + '.copying')
    shutil.copy2(source, temporary); os.replace(temporary, target)
    stats['copied'] = stats.get('copied', 0) + 1
    return False


def place(source, target, *, link):
    """Put a staged file at target atomically (a temporary name, then rename over the target): a hard link of the staged
    inode when link is true and possible, else a copy. The file previously at target is never written in place."""
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    if link and target.exists() and os.path.samefile(source, target):
        return True  # already in place (a resumed install); rename() between two links of one inode would do nothing
    temporary = target.with_name('.' + target.name + '.installing')
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    linked = False
    if link:
        try:
            os.link(source, temporary); linked = True
        except OSError:
            pass
    if not linked:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)
    return linked


def install_tree(source, target, links):
    """Install a staged dossier folder (shutil.copytree as before, folders included): files listed in links (paths
    relative to source) are hard-linked to the staged inode, which is the run's record; the rest are copied."""
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Commit destination already exists')
    for folder, dirs, files in os.walk(source):
        here = Path(folder); relative = here.relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            rel = (relative / name).as_posix()
            if rel in links:
                try:
                    os.link(here / name, target / relative / name)
                    continue
                except OSError:
                    pass
            shutil.copy2(here / name, target / relative / name)


# ------------------------------------------------------------------ backup and rollback of an install
BANK_FILES = ('registry.json', 'active_registry.json', 'basic_valid_registry.json', 'CATALOG.md', 'SHA256SUMS',
              'BASIC_VALID_CATALOG.md', 'BASIC_VALID_SHA256SUMS', RENDER_INDEX)


def backup(bank, destination, *, manifests, replaced_renders, added_renders, new_registry_ids):
    """Save what an install replaces: the bank-level files (copies), the render files it replaces or removes (hard links of
    the current inodes, which the install never writes in place: it renames new files over them) and the lists needed to
    undo it. Returns presence.json's value."""
    bank, destination = Path(bank), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    present = {}
    for name in [*BANK_FILES, *manifests]:
        source = bank / name; present[name] = source.is_file()
        if source.is_file():
            target = destination / 'bank' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    renders = bank / 'renders'
    before = sorted(p.name for p in renders.iterdir()) if renders.is_dir() else []
    for name in replaced_renders:
        target = destination / 'renders' / name; target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(renders / name, target)
        except OSError:
            shutil.copy2(renders / name, target)
    value = {'format': 2, 'bank_files': present, 'renders_before': before, 'renders_saved': sorted(replaced_renders),
             'renders_added': sorted(added_renders), 'new_registry_ids': list(new_registry_ids)}
    atomic_write_json(destination / 'presence.json', value)
    return value


def rollback(bank, saved):
    """Undo an install from its backup: remove what it added, restore (as copies) what it replaced, and check that the
    render folder holds exactly the files it held before."""
    bank, saved = Path(bank), Path(saved)
    presence = read_json(saved / 'presence.json')
    if presence.get('format') != 2:
        raise ValueError('Not a backup of this commit code')
    renders = bank / 'renders'
    for name in presence['renders_added']:
        (renders / name).unlink(missing_ok=True)
    for name in presence['renders_saved']:
        place(saved / 'renders' / name, renders / name, link=False)
    for name, existed in presence['bank_files'].items():
        if existed:
            place(saved / 'bank' / name, bank / name, link=False)
        else:
            (bank / name).unlink(missing_ok=True)
    for rid in presence['new_registry_ids']:
        (bank / 'tasks' / f'{rid}.json').unlink(missing_ok=True)
        shutil.rmtree(bank / 'dossiers' / rid, ignore_errors=True)
    for folder in (bank, renders, bank / 'commit_manifests', bank / 'tasks'):
        if folder.is_dir():
            for stray in folder.glob('.*.installing'):
                stray.unlink(missing_ok=True)
    after = sorted(p.name for p in renders.iterdir()) if renders.is_dir() else []
    if after != presence['renders_before']:
        raise ValueError('Rollback left the render folder different from before the install')
    return presence


__all__ = ['RENDER_INDEX', 'SHARED_RENDER_FILES', 'backup', 'contact_sheet', 'draw_renders', 'file_sha256',
           'format_render_index', 'hash_files', 'install_tree', 'link_or_copy', 'linkable', 'load_registry', 'patched',
           'place', 'plan_renders', 'read_render_index', 'rollback', 'shallow_dossier_check', 'thumbnail_size']
