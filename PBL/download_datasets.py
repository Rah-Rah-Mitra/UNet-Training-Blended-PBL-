#!/usr/bin/env python3
"""Download the image datasets for the PBL super-resolution notebook (standard library only).

The script needs nothing but Python >= 3.10, so it can run before any dependency is
installed (no ``requests``, no PIL). By default it fills ``data/`` next to this file, i.e.
next to ``SimplifiedUNetSR.ipynb``::

    data/
      BSD300/train/*.jpg    200 images, BSDS300 train split (original names, e.g. 100075.jpg)
      BSD300/test/*.jpg     100 images, BSDS300 test split
      SET14/test/*.png       14 images (baboon.png, barbara.png, ... zebra.png)
      ICDAR2003/train/*     258 images, ICDAR 2003 Robust Reading "SceneTrialTrain"
      ICDAR2003/test/*      251 images, "SceneTrialTest" (the paper reports 249)
      PROVENANCE.json       per dataset: source, URLs, UTC time, counts, verification, licence

Image files are stored byte for byte as downloaded; nothing is decoded or re-encoded.

Command line::

    python download_datasets.py                           # every dataset
    python download_datasets.py --datasets bsd300 set14   # a subset (case-insensitive)
    python download_datasets.py --root "D:/My Data"       # another data folder
    python download_datasets.py --force                   # re-download; old copy kept until success
    python download_datasets.py --verify-only             # recount and re-hash; no downloads
    python download_datasets.py --list-sources            # the ordered sources of each dataset
    python download_datasets.py --timeout 60              # network timeout in seconds (default 30)

The exit code is 0 when every requested dataset is ready and 1 when at least one failed.
A summary table is printed at the end.

From Python (this is what the notebook does)::

    import download_datasets as dd
    paths = dd.ensure_datasets(["BSD300", "SET14"], root="data")  # {"BSD300": Path(...), ...}

``ensure_datasets`` raises ``DatasetUnavailableError`` when every source of a dataset failed.
It processes the other datasets first. The error message explains how to download the data by hand.

How it works:

* Datasets that are already complete are skipped without any network access.
* Each dataset has an ordered list of sources (``--list-sources``). Any failure moves on to
  the next source: a network or HTTP error, a timeout, a checksum mismatch, a bad archive or
  a wrong image count.
* Every URL is tried up to 3 times with exponential backoff, unless the error is permanent
  (e.g. HTTP 403/404). urllib uses the standard proxy variables (``HTTPS_PROXY``,
  ``NO_PROXY``, ...). TLS certificates are always verified.
* Downloads are streamed to a temporary file in ``<root>/.downloads/`` and moved into place
  when complete. Archives are extracted into a temporary folder there. A dataset folder is
  created or replaced only after the new copy has been fully verified, so a failed attempt
  leaves nothing behind.
* Drop-in archives: a file saved as ``<root>/.downloads/<name>`` is used before any network
  source, after checksum verification where a checksum is known. The names are
  ``BSDS300-images.tgz``, ``Set14_HR.tar.gz``, and ``icdar2003_train.zip`` with
  ``icdar2003_test.zip``. Use them when a firewall blocks every source.
* ``dataset_manifest.json`` next to this file holds the SHA-256 of every mirrored image. If it
  is missing, the script still works with reduced verification and prints a warning.

Testing hook: the hidden option ``--source-override SLOT=URL[#sha256=HEX]`` is for testing
only. It is repeatable and does not appear in ``--help``. It points one source slot at
another URL, optionally with a different expected archive checksum. ``--list-sources`` shows
the slot names.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import gzip
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import socket
import ssl
import stat
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, TextIO

if sys.version_info < (3, 10):
    raise SystemExit("download_datasets.py needs Python 3.10 or newer.")

__all__ = [
    "DATASETS",
    "EXPECTED_COUNTS",
    "DatasetUnavailableError",
    "default_root",
    "dataset_dir",
    "is_complete",
    "ensure_datasets",
    "verify",
    "main",
]

# ----------------------------------------------------------------------------- public constants

DATASETS = ("BSD300", "SET14", "ICDAR2003")
EXPECTED_COUNTS = {
    "BSD300": {"train": 200, "test": 100},
    "SET14": {"test": 14},
    "ICDAR2003": {"train": 258, "test": 251},
}

# ----------------------------------------------------------------------------- configuration

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
MANIFEST_PATH = Path(__file__).resolve().parent / "dataset_manifest.json"
PROVENANCE_NAME = "PROVENANCE.json"
DOWNLOADS_DIRNAME = ".downloads"

USER_AGENT = (
    f"pbl-download-datasets/1.0 (Python {sys.version_info[0]}.{sys.version_info[1]} urllib)"
)
HTTP_ATTEMPTS = 3
RETRY_BACKOFF = (1.0, 2.0, 4.0)  # wait (s) after failed attempt 1, 2, 3; 3 attempts use 1 s and 2 s
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
CHUNK_SIZE = 1 << 16
MIRROR_WORKERS = 8  # parallel requests for the one-request-per-file GitHub mirrors

LICENCE_NOTES = {
    "BSD300": (
        "Berkeley Segmentation Dataset \u2014 free for non-commercial research and educational "
        "purposes (see https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)"
    ),
    "SET14": (
        "Set14 benchmark (Zeyde et al. 2010); classic test images used for research benchmarking"
    ),
    "ICDAR2003": (
        "ICDAR 2003 Robust Reading Competition data (Lucas et al.), distributed by IAPR-TC11 "
        "for research"
    ),
}

_BSDS_TARBALL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
_BSDS_TARBALL_HTTP = (
    "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
)
_BSDS_MIRROR = (
    "https://raw.githubusercontent.com/BUPTLdy/pytorch-lapsrn/"
    "6948718c86c9ff722954146c7ecbea15f00033c9/dataset/BSDS300"
)
_SET14_MIRROR = (
    "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/"
    "8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/data/Set14/image_SRF_2"
)
_SET14_TARBALL = (
    "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz"
)
_ICDAR_IAPR = "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading"
_ICDAR_ESSEX = "http://algoval.essex.ac.uk/icdar/datasets"
_ICDAR_TRAIN_SHA256 = "9d86df514eb09dd693fb0b8c671ef54a0cfe02e803b1bbef9fc676061502eb94"  # docTR
_ICDAR_TEST_SHA256 = "dbc4b5fd5d04616b8464a1b42ea22db351ee22c2546dd15ac35611857ea111f8"  # docTR

# Set14 in canonical order; the SelfExSR mirror stores them as img_001 ... img_014.
_SET14_NAMES = (
    "baboon", "barbara", "bridge", "coastguard", "comic", "face", "flowers",
    "foreman", "lenna", "man", "monarch", "pepper", "ppt3", "zebra",
)  # fmt: skip

_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")
_SAFE_RELATIVE = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-][A-Za-z0-9._-]*")  # "<split>/<file>"
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_TAR_DATA_FILTER = hasattr(tarfile, "data_filter")  # Python >= 3.12 and the security backports

_LICENCES_SHOWN: set[str] = set()  # licence notes are printed once per process
_OVERRIDES: dict[str, tuple[str, str | None]] = {}  # testing hook: slot -> (url, sha256)

# ----------------------------------------------------------------------------- errors


class DatasetUnavailableError(RuntimeError):
    """Every source failed for at least one dataset.

    The message gives the reason for each failed source and manual-download instructions:
    which URL to fetch and where to put or unpack it. ``failures`` maps each failed dataset to
    its per-source errors. ``available`` maps the datasets that are ready to their folders.
    """

    def __init__(
        self,
        message: str,
        failures: dict[str, list[str]] | None = None,
        available: dict[str, Path] | None = None,
    ) -> None:
        super().__init__(message)
        self.failures = dict(failures or {})
        self.available = dict(available or {})


class _SourceError(Exception):
    """A source failed; the message is the one-line reason shown to the user."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _UnsafeArchiveError(Exception):
    """An archive member has an absolute path, escapes the target folder, or is a link/device."""


# Errors that mean "refused for safety": our own guard, plus tarfile's "data" filter if present.
_UNSAFE_ARCHIVE_ERRORS: tuple[type[Exception], ...] = (_UnsafeArchiveError,) + (
    (tarfile.FilterError,) if hasattr(tarfile, "FilterError") else ()
)


# ----------------------------------------------------------------------------- console output


def _write(stream: TextIO, text: str) -> None:
    """Write and flush, replacing characters the stream's encoding cannot represent."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding))
    stream.flush()


def _warn(message: str) -> None:
    """Print a warning to stderr (also when quiet)."""
    _write(sys.stderr, f"WARNING: {message}\n")


def _is_interactive(stream: TextIO) -> bool:
    """True for terminals and Jupyter, where a self-overwriting progress line renders well."""
    if "ipykernel" in sys.modules:
        return True
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


class _Reporter:
    """Console output for one run.

    ``info`` lines and the progress line are suppressed when quiet; warnings always go to
    stderr. The progress line redraws itself with ``\\r`` only on terminals and in Jupyter,
    so captured logs stay clean.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._out: TextIO = sys.stdout
        self._live = not quiet and _is_interactive(self._out)
        self._lock = threading.Lock()
        self._progress_width = 0  # length of the progress line on screen (0: none)
        self._last_draw = 0.0

    def info(self, message: str) -> None:
        """Print a normal message unless quiet."""
        if not self.quiet:
            self._line(self._out, message)

    def warn(self, message: str) -> None:
        """Print a warning to stderr, even when quiet."""
        self._line(sys.stderr, f"WARNING: {message}")

    def progress(self, text: str) -> None:
        """Redraw the transient progress line (at most five times per second)."""
        now = time.monotonic()
        if not self._live or now - self._last_draw < 0.2:
            return
        with self._lock:
            self._last_draw = now
            _write(self._out, "\r" + text.ljust(self._progress_width))
            self._progress_width = len(text)

    def _line(self, stream: TextIO, message: str) -> None:
        with self._lock:
            if self._progress_width:  # wipe the progress line before printing a real one
                _write(self._out, "\r" + " " * self._progress_width + "\r")
                self._progress_width = 0
            _write(stream, message + "\n")


def _size(num_bytes: float) -> str:
    """Human-readable byte count."""
    if num_bytes < 1_000_000:
        return f"{num_bytes / 1000:.0f} kB"
    return f"{num_bytes / 1_000_000:.1f} MB"


def _format_counts(counts: dict[str, int]) -> str:
    """{"train": 200, "test": 100} -> "train=200 test=100"."""
    return " ".join(f"{split}={count}" for split, count in counts.items())


# ----------------------------------------------------------------------------- source model


@dataclass(frozen=True)
class _Archive:
    """One archive file of an archive source."""

    slot: str  # name shown by --list-sources and accepted by --source-override
    url: str
    filename: str  # name of the drop-in copy in <root>/.downloads/
    sha256: str | None = None  # None: no published checksum
    split: str = ""  # split the archive provides ("" = all of them, found by the layout)


@dataclass(frozen=True)
class _MirrorFile:
    """One file of a per-file mirror."""

    dest: str  # "<split>/<file name>" inside the dataset folder
    url: str
    sha256: str | None  # None when the manifest is missing


@dataclass
class _Fetched:
    """Outcome of a successful fetch, for the console and PROVENANCE.json."""

    urls: list[str]
    verification: str
    warnings: list[str] = field(default_factory=list)
    deviations: dict[str, str] = field(default_factory=dict)  # "split/file" -> sha256 on disk
    note: str = ""  # remark about the source's data, copied into PROVENANCE.json


@dataclass
class _Context:
    """Settings shared by everything that runs for one ensure_datasets() call."""

    root: Path
    timeout: float
    report: _Reporter

    @property
    def downloads(self) -> Path:
        """``<root>/.downloads``: drop-in archives, temporary files and folders."""
        return self.root / DOWNLOADS_DIRNAME


# A layout maps an extracted archive to (split, extracted file, final file name) triples.
_Layout = Callable[[Path, str], Iterator[tuple[str, Path, str]]]


@dataclass(frozen=True)
class _ArchiveSource:
    """Archives that are downloaded (or taken from .downloads/), checked, extracted, normalised."""

    key: str
    label: str
    archives: tuple[_Archive, ...]
    layout: _Layout
    local: bool = False  # True: use the drop-in copies in <root>/.downloads/, never the network

    def slots(self) -> list[tuple[str, str, str]]:
        """(slot, URL, checksum note) of each archive, for --list-sources."""
        return [
            (a.slot, a.url, f"sha256 {a.sha256}" if a.sha256 else "no published checksum")
            for a in self.archives
        ]

    def fetch(self, spec: _DatasetSpec, ctx: _Context, work: Path) -> _Fetched:
        """Fill ``work/stage/<split>/`` with the normalised images of every archive."""
        stage = work / "stage"
        taken: dict[str, set[str]] = {}
        warnings: list[str] = []
        for archive in self.archives:
            path = self._obtain(archive, ctx, work)
            extracted = work / "extracted" / (archive.split or "all")
            _extract(path, extracted)
            _stage_images(self.layout(extracted, archive.split), stage, taken, warnings)

        checks = [f"{a.filename} sha256 verified" for a in self.archives if a.sha256]
        checked, deviations = _compare_with_manifest(spec.name, stage)
        if checked:
            checks.append(
                f"{checked - len(deviations)}/{checked} images match dataset_manifest.json"
            )
        if deviations:
            warnings.append(
                f"{len(deviations)} of {checked} images differ from the dataset_manifest.json "
                "sha256; kept, because this archive's bytes may legitimately differ from the "
                "mirror (recorded in PROVENANCE.json)"
            )
        status = "warning" if deviations else "verified" if checks else "counts only"
        verification = f"{status}: " + ("; ".join(checks) or "no published checksum")
        if self.local:
            urls = [(ctx.downloads / a.filename).as_uri() for a in self.archives]
        else:
            urls = [a.url for a in self.archives]
        return _Fetched(urls, verification, warnings, deviations)

    def _obtain(self, archive: _Archive, ctx: _Context, work: Path) -> Path:
        """Return the archive file: the drop-in copy for a local source, else a fresh download."""
        if self.local:
            path = ctx.downloads / archive.filename
            ctx.report.info(f"        using {path}")
            actual = _sha256_file(path) if archive.sha256 else None
        else:
            path = work / "archives" / archive.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            ctx.report.info(f"        GET {archive.url}")
            actual = _download(archive.url, path, ctx, progress_label=archive.filename).sha256
        if archive.sha256 and actual != archive.sha256:
            raise _SourceError(
                f"sha256 mismatch for {archive.filename} (expected {archive.sha256}, got {actual})"
            )
        return path


@dataclass(frozen=True)
class _MirrorSource:
    """A pinned GitHub mirror: one request per file, each file checked against the manifest."""

    key: str
    label: str
    slot: str
    base_url: str
    url_pattern: str  # how file URLs look, for messages; "{base}" stands for base_url
    list_files: Callable[[_Context, str, Path], list[_MirrorFile]]
    note: str = ""  # remark about this mirror's data, recorded in PROVENANCE.json

    def slots(self) -> list[tuple[str, str, str]]:
        """(slot, URL pattern, checksum note), for --list-sources."""
        return [(self.slot, self._pattern(), "sha256 of every file from dataset_manifest.json")]

    def fetch(self, spec: _DatasetSpec, ctx: _Context, work: Path) -> _Fetched:
        """Download every file into ``work/stage/<split>/``; a checksum mismatch fails it."""
        files = self.list_files(ctx, self.base_url, work)
        ctx.report.info(f"        GET {self._pattern()} ({len(files)} files)")
        _download_files(files, work / "stage", ctx)
        if files and all(f.sha256 for f in files):
            verification = f"verified: sha256 of all {len(files)} files match dataset_manifest.json"
        else:
            verification = "counts only: dataset_manifest.json missing, sha256 not verified"
        return _Fetched([self._pattern()], verification, note=self.note)

    def _pattern(self) -> str:
        return self.url_pattern.replace("{base}", self.base_url)


_Source = _ArchiveSource | _MirrorSource


@dataclass(frozen=True)
class _DatasetSpec:
    """Everything the downloader knows about one dataset."""

    name: str
    splits: dict[str, int]  # expected image count per split
    strict_counts: bool  # False: any non-zero count is accepted, with a warning if unexpected
    sources: tuple[_Source, ...]
    unpack_hint: str  # manual-unpacking advice; "{root}" is replaced by the data folder


@dataclass
class _Result:
    """What happened to one dataset, for the summary table and the error message."""

    name: str
    ok: bool
    status: str
    path: Path
    counts: dict[str, int] = field(default_factory=dict)
    source: str = ""
    errors: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- files and counts


def _is_image_name(name: str) -> bool:
    """Image extension (case-insensitive) and not a hidden/dot file."""
    return not name.startswith(".") and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def _count_images(folder: Path) -> int:
    """Number of image files directly inside folder (0 if it does not exist)."""
    try:
        with os.scandir(folder) as entries:
            return sum(1 for entry in entries if _is_image_name(entry.name) and entry.is_file())
    except (FileNotFoundError, NotADirectoryError):
        return 0


def _split_counts(folder: Path, splits: Iterable[str]) -> dict[str, int]:
    """Image count of every split folder of a dataset."""
    return {split: _count_images(folder / split) for split in splits}


def _count_problem(spec: _DatasetSpec, counts: dict[str, int]) -> str | None:
    """Why counts make a dataset incomplete, or None if they are acceptable."""
    expected = _format_counts(spec.splits)
    if spec.strict_counts and counts != spec.splits:
        return f"wrong image count: {_format_counts(counts)} (expected {expected})"
    if not all(counts.values()):
        return f"no images in some split: {_format_counts(counts)} (expected {expected})"
    return None


def _count_note(spec: _DatasetSpec, counts: dict[str, int]) -> str | None:
    """Warning for accepted counts that differ from the expected ones (ICDAR2003 only)."""
    if counts == spec.splits:
        return None
    expected = _format_counts(spec.splits)
    return f"image counts {_format_counts(counts)} differ from the expected {expected} (accepted)"


def _mismatch(what: str, expected: str, actual: str) -> str:
    """One-line checksum mismatch message with shortened hashes."""
    return f"sha256 mismatch for {what} (expected {expected[:16]}..., got {actual[:16]}...)"


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rmtree(path: Path) -> None:
    """Delete a folder tree (best effort), making read-only files writable first (Windows)."""

    def make_writable_and_retry(
        function: Callable[[str], object], target: str, _error: object
    ) -> None:
        os.chmod(target, stat.S_IWRITE)
        function(target)

    if not os.path.lexists(path):
        return
    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=make_writable_and_retry)
        else:
            shutil.rmtree(path, onerror=make_writable_and_retry)
    except OSError as exc:
        _warn(f"could not remove the temporary folder {path}: {exc}")


def _remove_if_empty(folder: Path) -> None:
    """Remove folder if it exists and is empty."""
    with contextlib.suppress(OSError):
        folder.rmdir()


# ------------------------------------------------------------------------ manifest and provenance


@cache
def _load_manifest() -> dict[str, Any] | None:
    """dataset_manifest.json next to this script, or None (with one warning) if unusable."""
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("the top level is not a JSON object")
        return data
    except FileNotFoundError:
        problem = "not found"
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        problem = f"unreadable ({exc})"
    _warn(
        f"{MANIFEST_PATH} {problem}: per-file SHA-256 checks are disabled (reduced verification); "
        "the GitHub mirrors use the BSDS iids_*.txt lists and the built-in Set14 names instead"
    )
    return None


def _manifest_files(name: str) -> dict[str, tuple[str, str | None]] | None:
    """Manifest entries of a dataset as {"split/file": (sha256, mirror file name or None)}.

    Returns None when the manifest is missing or has no files for this dataset. Entries that
    are not images of an expected split (such as the BSDS ``iids_*.txt`` lists) are ignored.
    """
    manifest = _load_manifest()
    section = manifest.get(name) if manifest else None
    files = section.get("files") if isinstance(section, dict) else None
    if not isinstance(files, dict):
        return None
    entries: dict[str, tuple[str, str | None]] = {}
    for rel, value in files.items():
        if not (_SAFE_RELATIVE.fullmatch(rel) and rel.split("/")[0] in EXPECTED_COUNTS[name]):
            continue
        if isinstance(value, dict):  # {"source": "<file on the mirror>", "sha256": "..."}
            sha256, source = value.get("sha256"), value.get("source")
        else:  # plain "<sha256>"
            sha256, source = value, None
        valid_source = source is None or (isinstance(source, str) and _SAFE_NAME.fullmatch(source))
        if not (isinstance(sha256, str) and _SHA256_HEX.fullmatch(sha256) and valid_source):
            _warn(f"ignoring the malformed {MANIFEST_PATH.name} entry {name}: {rel}")
            continue
        entries[rel] = (sha256.lower(), source)
    return entries or None


def _compare_with_manifest(name: str, folder: Path) -> tuple[int, dict[str, str]]:
    """Hash the files in folder that have a manifest entry.

    Returns the number of files checked and {"split/file": actual sha256} for those that differ.
    Manifest entries without a file on disk are skipped (the image counts cover completeness).
    """
    checked, deviations = 0, {}
    for rel, (expected, _source) in sorted((_manifest_files(name) or {}).items()):
        path = folder / rel
        if path.is_file():
            checked += 1
            actual = _sha256_file(path)
            if actual != expected:
                deviations[rel] = actual
    return checked, deviations


def _read_provenance(root: Path) -> dict[str, Any]:
    """Contents of <root>/PROVENANCE.json ({} if missing or unreadable)."""
    path = root / PROVENANCE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        _warn(f"ignoring unreadable {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _known_deviations(root: Path, name: str) -> dict[str, str]:
    """Manifest deviations accepted at download time (recorded in PROVENANCE.json)."""
    entry = _read_provenance(root).get(name)
    deviations = entry.get("manifest_deviations") if isinstance(entry, dict) else None
    if not isinstance(deviations, dict):
        return {}
    return {k: v for k, v in deviations.items() if isinstance(k, str) and isinstance(v, str)}


def _dataset_order(name: str) -> int:
    return DATASETS.index(name) if name in DATASETS else len(DATASETS)


def _write_provenance(ctx: _Context, name: str, entry: dict[str, Any]) -> None:
    """Add or replace one dataset's entry in <root>/PROVENANCE.json (atomic write)."""
    path = ctx.root / PROVENANCE_NAME
    data = _read_provenance(ctx.root)
    data[name] = entry
    ordered = dict(sorted(data.items(), key=lambda item: _dataset_order(item[0])))
    tmp = ctx.root / f".{PROVENANCE_NAME}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        ctx.report.warn(f"could not write {path}: {exc}")
        with contextlib.suppress(OSError):
            tmp.unlink()


# ----------------------------------------------------------------------------- HTTP


@dataclass(frozen=True)
class _Download:
    size: int
    sha256: str


def _proxy_for(url: str) -> str | None:
    """The proxy urllib uses for url, if any (only used to word error messages)."""
    parts = urllib.parse.urlsplit(url)
    try:
        proxies = urllib.request.getproxies()
        if parts.hostname and proxies and urllib.request.proxy_bypass(parts.hostname):
            return None
    except (OSError, ValueError):
        return None
    return proxies.get(parts.scheme)


def _explain_error(exc: BaseException, url: str, timeout: float) -> tuple[str, bool]:
    """One-line reason for a failed request, and whether another attempt may help."""
    host = urllib.parse.urlsplit(url).hostname or url
    if isinstance(exc, _SourceError):
        return str(exc), exc.retryable
    if isinstance(exc, urllib.error.HTTPError):
        reason = f"HTTP {exc.code} {exc.reason or ''}".rstrip()
        if exc.code in (401, 403, 407):
            reason += " (access denied by the server or a proxy/firewall)"
        return reason, exc.code in RETRYABLE_HTTP_STATUS
    cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(cause, TimeoutError):  # socket.timeout is TimeoutError since Python 3.10
        return f"timed out after {timeout:g}s", True
    if isinstance(cause, ssl.SSLCertVerificationError):
        return (
            f"TLS certificate verification failed for {host} ({cause.verify_message}); behind an "
            "inspecting proxy, set SSL_CERT_FILE to its CA bundle (verification stays enabled)",
            False,
        )
    if isinstance(cause, socket.gaierror):  # only "temporary failure" is worth another attempt
        transient = cause.errno == getattr(socket, "EAI_AGAIN", None)
        return f"DNS lookup failed for {host} ({cause.strerror})", transient
    tunnel = re.search(r"Tunnel connection failed: (\d{3})", str(cause))  # HTTPS via a proxy
    if tunnel:
        code = int(tunnel.group(1))
        reason = f"HTTP {code} from proxy/firewall (CONNECT to {host} refused)"
        return reason, code in RETRYABLE_HTTP_STATUS
    if isinstance(cause, ConnectionRefusedError):
        proxy = _proxy_for(url)
        return f"connection refused by {f'proxy {proxy}' if proxy else host}", True
    if isinstance(cause, OSError) and (
        cause.filename or cause.errno in (errno.ENOSPC, errno.EROFS, getattr(errno, "EDQUOT", -1))
    ):
        return f"local file error: {cause}", False
    if isinstance(cause, (OSError, http.client.HTTPException)):  # resets, TLS and protocol errors
        return f"network error: {cause or type(cause).__name__}", True
    if isinstance(cause, ValueError):
        return f"invalid URL {url!r} ({cause})", False
    if isinstance(cause, str):
        return cause, False
    return f"{type(cause).__name__}: {cause}", False


def _download_once(url: str, dest: Path, ctx: _Context, progress_label: str | None) -> _Download:
    """Stream url into a temporary file in <root>/.downloads/, then os.replace it to dest."""
    ctx.downloads.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="download-", suffix=".part", dir=ctx.downloads)
    tmp = Path(tmp_name)
    digest, size, started = hashlib.sha256(), 0, time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(request, timeout=ctx.timeout) as response,
        ):
            if response.headers.get_content_type() == "text/html":
                raise _SourceError("got an HTML page instead of the file (login or error page?)")
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            while chunk := response.read(CHUNK_SIZE):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if progress_label:
                    shown = (
                        f"{_size(size)} / {_size(total)} ({100 * size // total}%)"
                        if total
                        else _size(size)
                    )
                    ctx.report.progress(f"        {progress_label}: {shown}")
        if total is not None and size != total:
            raise _SourceError(
                f"connection closed early ({_size(size)} of {_size(total)})", retryable=True
            )
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if progress_label:
        ctx.report.info(
            f"        {progress_label}: {_size(size)} in {time.monotonic() - started:.1f}s"
        )
    return _Download(size, digest.hexdigest())


def _download(url: str, dest: Path, ctx: _Context, progress_label: str | None = None) -> _Download:
    """Download url to dest, retrying transient failures with exponential backoff.

    Raises _SourceError with a one-line reason when the URL cannot be fetched.
    """
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            return _download_once(url, dest, ctx, progress_label)
        except Exception as exc:  # classified below; KeyboardInterrupt is not caught
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()  # an error response still holds its connection
            reason, retryable = _explain_error(exc, url, ctx.timeout)
            if not retryable or attempt == HTTP_ATTEMPTS:
                raise _SourceError(
                    reason + (f" (after {attempt} attempts)" if attempt > 1 else "")
                ) from exc
            delay = RETRY_BACKOFF[attempt - 1]
            ctx.report.info(
                f"        {reason}; attempt {attempt + 1}/{HTTP_ATTEMPTS} in {delay:g}s: {url}"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _download_files(files: Sequence[_MirrorFile], stage: Path, ctx: _Context) -> None:
    """Fetch mirror files in parallel into stage, verifying each checksum (mismatch = failure)."""

    def fetch_one(item: _MirrorFile) -> int:
        dest = stage / item.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = _download(item.url, dest, ctx)
        except _SourceError as exc:
            raise _SourceError(f"{item.dest}: {exc}") from exc
        if item.sha256 and result.sha256 != item.sha256:
            raise _SourceError(_mismatch(item.dest, item.sha256, result.sha256))
        return result.size

    started, done, total = time.monotonic(), 0, 0
    with ThreadPoolExecutor(max_workers=MIRROR_WORKERS) as pool:
        futures = [pool.submit(fetch_one, item) for item in files]
        try:
            for future in as_completed(futures):
                total += future.result()
                done += 1
                ctx.report.progress(f"        {done}/{len(files)} files, {_size(total)}")
        except BaseException:
            for future in futures:  # stop queued downloads; running ones finish before cleanup
                future.cancel()
            raise
    ctx.report.info(f"        {done} files, {_size(total)} in {time.monotonic() - started:.1f}s")


# ----------------------------------------------------------------------------- archives


def _check_member_path(name: str, dest: Path) -> None:
    """Refuse absolute paths, drive letters and '..' (zip-slip / tar path traversal)."""
    posix = name.replace("\\", "/")
    if posix.startswith("/") or re.match(r"[A-Za-z]:", posix):
        raise _UnsafeArchiveError(f"absolute path {name!r}")
    if ".." in posix.split("/"):
        raise _UnsafeArchiveError(f"path traversal {name!r}")
    root = dest.resolve()
    if not (root / posix).resolve().is_relative_to(root):
        raise _UnsafeArchiveError(f"{name!r} would be written outside the extraction folder")


def _extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip after checking every member name (zip-slip guard) and refusing symlinks."""
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for info in members:
            _check_member_path(info.filename, dest)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise _UnsafeArchiveError(f"symbolic link {info.filename!r}")
        zf.extractall(dest, members)


def _extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tar archive with the "data" filter, or with a manual guard on older Pythons."""
    with tarfile.open(archive, "r:*") as tar:
        if _TAR_DATA_FILTER:
            tar.extractall(dest, filter="data")
            return
        members = tar.getmembers()
        for member in members:  # validate everything before writing anything
            _check_member_path(member.name, dest)
            if member.issym() or member.islnk():
                raise _UnsafeArchiveError(f"link {member.name!r}")
            if member.isdev():
                raise _UnsafeArchiveError(f"device or FIFO {member.name!r}")
        for member in members:  # only regular files and folders remain
            target = dest / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)


def _extract(archive: Path, dest: Path) -> None:
    """Extract a .zip or tar archive into dest; problems become a one-line _SourceError."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if archive.name.lower().endswith(".zip"):
            _extract_zip(archive, dest)
        else:
            _extract_tar(archive, dest)
    except _UNSAFE_ARCHIVE_ERRORS as exc:
        raise _SourceError(f"unsafe archive {archive.name} refused: {exc}") from exc
    except (tarfile.TarError, zipfile.BadZipFile, gzip.BadGzipFile, zlib.error, EOFError) as exc:
        detail = str(exc).splitlines()[0].rstrip(": ") if str(exc) else type(exc).__name__
        with open(archive, "rb") as handle:
            looks_like_html = handle.read(512).lstrip().lower().startswith((b"<!doctype", b"<html"))
        hint = " (the file is a web page, not an archive)" if looks_like_html else ""
        raise _SourceError(f"bad archive {archive.name}: {detail}{hint}") from exc


def _image_files(top: Path) -> Iterator[Path]:
    """Image files below top in a stable order, skipping __MACOSX, dot-files and links."""
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = sorted(d for d in dirnames if d != "__MACOSX" and not d.startswith("."))
        for filename in sorted(filenames):
            path = Path(dirpath, filename)
            if _is_image_name(filename) and not path.is_symlink() and path.is_file():
                yield path


def _layout_bsds(extracted: Path, _split: str) -> Iterator[tuple[str, Path, str]]:
    """BSDS300 tarball: ``BSDS300/images/{train,test}/<id>.jpg`` keep their names."""
    for path in _image_files(extracted):
        if path.parent.name in ("train", "test") and path.parent.parent.name == "images":
            yield path.parent.name, path, path.name


def _layout_flat(extracted: Path, split: str) -> Iterator[tuple[str, Path, str]]:
    """Unknown inner layout (Set14 tarball): every image goes to split under its base name."""
    for path in _image_files(extracted):
        yield split, path, path.name


def _layout_icdar(extracted: Path, split: str) -> Iterator[tuple[str, Path, str]]:
    """ICDAR 2003 zip: ``<top>/<dir>/<image>`` becomes ``<dir>_<image>`` in [A-Za-z0-9._-]."""
    for path in _image_files(extracted):
        name = path.name if path.parent == extracted else f"{path.parent.name}_{path.name}"
        yield split, path, re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _stage_images(
    items: Iterable[tuple[str, Path, str]],
    stage: Path,
    taken: dict[str, set[str]],
    warnings: list[str],
) -> None:
    """Move normalised images into stage/<split>/<name>, renaming case-insensitive duplicates."""
    for split, path, name in items:
        used = taken.setdefault(split, set())
        stem, suffix = os.path.splitext(name)
        final, number = name, 1
        while final.lower() in used:
            number += 1
            final = f"{stem}-{number}{suffix}"
        if final != name:
            warnings.append(f"duplicate file name {split}/{name} stored as {final}")
        used.add(final.lower())
        (stage / split).mkdir(parents=True, exist_ok=True)
        os.replace(path, stage / split / final)


# ----------------------------------------------------------------------------- mirror file lists


def _bsd300_mirror_files(ctx: _Context, base: str, work: Path) -> list[_MirrorFile]:
    """BSDS300 files on the mirror: names and hashes from the manifest, else from iids_*.txt."""
    entries = _manifest_files("BSD300")
    if entries:
        return [
            _MirrorFile(rel, f"{base}/images/{rel}", sha)
            for rel, (sha, _) in sorted(entries.items())
        ]
    files = []
    for split in EXPECTED_COUNTS["BSD300"]:
        listing = work / f"iids_{split}.txt"
        _download(f"{base}/iids_{split}.txt", listing, ctx)
        image_ids = [
            i for i in listing.read_text(encoding="ascii", errors="replace").split() if i.isdigit()
        ]
        files += [
            _MirrorFile(f"{split}/{i}.jpg", f"{base}/images/{split}/{i}.jpg", None)
            for i in image_ids
        ]
    return files


def _set14_mirror_files(_ctx: _Context, base: str, _work: Path) -> list[_MirrorFile]:
    """Set14 files on the mirror, saved under canonical names (manifest, else built-in list)."""
    entries = _manifest_files("SET14")
    if entries and all(source for _, source in entries.values()):
        return [
            _MirrorFile(rel, f"{base}/{source}", sha)
            for rel, (sha, source) in sorted(entries.items())
        ]
    return [
        _MirrorFile(f"test/{name}.png", f"{base}/img_{number:03d}_SRF_2_HR.png", None)
        for number, name in enumerate(_SET14_NAMES, 1)
    ]


# ----------------------------------------------------------------------------- dataset registry

_SPECS: dict[str, _DatasetSpec] = {
    "BSD300": _DatasetSpec(
        name="BSD300",
        splits=EXPECTED_COUNTS["BSD300"],
        strict_counts=True,
        sources=(
            _ArchiveSource(
                "official-https",
                "official BSDS300 tarball (https)",
                (_Archive("bsd300-https", _BSDS_TARBALL, "BSDS300-images.tgz"),),
                _layout_bsds,
            ),
            _ArchiveSource(
                "official-http",
                "official BSDS300 tarball (plain http)",
                (_Archive("bsd300-http", _BSDS_TARBALL_HTTP, "BSDS300-images.tgz"),),
                _layout_bsds,
            ),
            _MirrorSource(
                "github-mirror",
                "GitHub mirror BUPTLdy/pytorch-lapsrn @ 6948718 (one request per file)",
                "bsd300-mirror",
                _BSDS_MIRROR,
                "{base}/images/<split>/<file>",
                _bsd300_mirror_files,
            ),
        ),
        unpack_hint=(
            "(Or unpack it yourself so that BSDS300/images/train/*.jpg end up in "
            "{root}/BSD300/train/ and BSDS300/images/test/*.jpg in {root}/BSD300/test/.)"
        ),
    ),
    "SET14": _DatasetSpec(
        name="SET14",
        splits=EXPECTED_COUNTS["SET14"],
        strict_counts=True,
        sources=(
            _MirrorSource(
                "github-mirror",
                "GitHub mirror jbhuang0604/SelfExSR @ 8f6dd8c (one request per file)",
                "set14-mirror",
                _SET14_MIRROR,
                "{base}/img_NNN_SRF_2_HR.png",
                _set14_mirror_files,
                note=(
                    "these HR images are mod-cropped to a multiple of 2: comic, ppt3 and zebra "
                    "lose at most one pixel row or column compared with the original Set14 images"
                ),
            ),
            _ArchiveSource(
                "huggingface",
                "Hugging Face eugenesiow/Set14 (Set14_HR.tar.gz)",
                (_Archive("set14-hf", _SET14_TARBALL, "Set14_HR.tar.gz", split="test"),),
                _layout_flat,
            ),
        ),
        unpack_hint="(Or unpack the 14 images it contains into {root}/SET14/test/.)",
    ),
    "ICDAR2003": _DatasetSpec(
        name="ICDAR2003",
        splits=EXPECTED_COUNTS["ICDAR2003"],
        strict_counts=False,
        sources=(
            _ArchiveSource(
                "iapr-tc11",
                "official IAPR-TC11 host (plain http)",
                (
                    _Archive(
                        "icdar2003-iapr-train",
                        f"{_ICDAR_IAPR}/TrialTrain/scene.zip",
                        "icdar2003_train.zip",
                        _ICDAR_TRAIN_SHA256,
                        "train",
                    ),
                    _Archive(
                        "icdar2003-iapr-test",
                        f"{_ICDAR_IAPR}/TrialTest/scene.zip",
                        "icdar2003_test.zip",
                        _ICDAR_TEST_SHA256,
                        "test",
                    ),
                ),
                _layout_icdar,
            ),
            _ArchiveSource(
                "essex",
                "fallback host algoval.essex.ac.uk (plain http)",
                (
                    _Archive(
                        "icdar2003-essex-train",
                        f"{_ICDAR_ESSEX}/TrialTrain/scene.zip",
                        "icdar2003_train.zip",
                        _ICDAR_TRAIN_SHA256,
                        "train",
                    ),
                    _Archive(
                        "icdar2003-essex-test",
                        f"{_ICDAR_ESSEX}/TrialTest/scene.zip",
                        "icdar2003_test.zip",
                        _ICDAR_TEST_SHA256,
                        "test",
                    ),
                ),
                _layout_icdar,
            ),
        ),
        unpack_hint=(
            "(Both zips are needed; the script verifies them and unpacks the images into "
            "{root}/ICDAR2003/train/ and {root}/ICDAR2003/test/.)"
        ),
    ),
}


def _spec(name: str) -> _DatasetSpec:
    """Look up a dataset by case-insensitive name."""
    key = str(name).strip().upper()
    if key not in _SPECS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {', '.join(DATASETS)}")
    return _SPECS[key]


def _canonical_names(names: str | Iterable[str]) -> list[str]:
    """Canonical, de-duplicated dataset names; "all" expands to every dataset."""
    if isinstance(names, str):
        names = [names]
    result: list[str] = []
    for name in names:
        expanded = DATASETS if str(name).strip().lower() == "all" else (_spec(name).name,)
        result += [n for n in expanded if n not in result]
    return result


def _all_slots() -> set[str]:
    return {
        slot for spec in _SPECS.values() for source in spec.sources for slot, _, _ in source.slots()
    }


def _set_source_overrides(items: Iterable[str]) -> None:
    """Testing hook behind --source-override: ``SLOT=URL[#sha256=HEX]`` per item.

    A mirror slot takes the base URL that file paths are appended to. ``#sha256=`` replaces
    the archive checksum the download must match. Replaces any earlier overrides.
    """
    parsed: dict[str, tuple[str, str | None]] = {}
    known = _all_slots()
    for item in items:
        slot, _, value = item.partition("=")
        slot = slot.strip().lower()
        if slot not in known:
            raise ValueError(
                f"unknown source slot {slot!r} in {item!r}; known: {', '.join(sorted(known))}"
            )
        url, _, fragment = value.strip().partition("#")
        sha256 = fragment.removeprefix("sha256=") if fragment else None
        if not url or (sha256 is not None and not _SHA256_HEX.fullmatch(sha256)):
            raise ValueError(f"expected SLOT=URL or SLOT=URL#sha256=<64 hex digits>, got {item!r}")
        parsed[slot] = (url, sha256.lower() if sha256 else None)
    _OVERRIDES.clear()
    _OVERRIDES.update(parsed)


def _with_overrides(source: _Source) -> _Source:
    """source with any --source-override applied (testing only)."""
    if isinstance(source, _MirrorSource):
        if source.slot not in _OVERRIDES:
            return source
        return dataclasses.replace(source, base_url=_OVERRIDES[source.slot][0].rstrip("/"))

    def override(archive: _Archive) -> _Archive:
        if archive.slot not in _OVERRIDES:
            return archive
        url, sha256 = _OVERRIDES[archive.slot]
        return dataclasses.replace(archive, url=url, sha256=sha256 or archive.sha256)

    return dataclasses.replace(source, archives=tuple(override(a) for a in source.archives))


def _plan_sources(spec: _DatasetSpec, downloads: Path) -> list[_Source]:
    """Sources in the order they are tried: complete drop-in archives first, then the network."""
    sources = [_with_overrides(source) for source in spec.sources]
    for source in sources:
        if not isinstance(source, _ArchiveSource):
            continue
        if all((downloads / a.filename).is_file() for a in source.archives):
            files = " + ".join(a.filename for a in source.archives)
            local = dataclasses.replace(
                source,
                key="local-archive",
                label=f"drop-in archive in {downloads}: {files}",
                local=True,
            )
            return [local, *sources]
    return sources


# -------------------------------------------------------------------------- downloading a dataset


def _install(stage: Path, dest: Path, work: Path) -> None:
    """Move the verified staging folder to dest; an existing dest is replaced only now."""
    previous = work / "previous"
    if os.path.lexists(dest):
        os.replace(dest, previous)  # dropped together with the work folder
    try:
        os.replace(stage, dest)
    except OSError:
        if os.path.lexists(previous):
            os.replace(previous, dest)
        raise


def _attempt(
    spec: _DatasetSpec, source: _Source, ctx: _Context, dest: Path
) -> tuple[_Fetched, dict[str, int]]:
    """Run one source in a fresh temporary folder and install the result if it is complete.

    The temporary folder (downloads, extracted files, staged images) is always deleted.
    """
    ctx.downloads.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{spec.name}-", suffix=".tmp", dir=ctx.downloads))
    try:
        stage = work / "stage"
        for split in spec.splits:
            (stage / split).mkdir(parents=True)
        fetched = source.fetch(spec, ctx, work)
        counts = _split_counts(stage, spec.splits)
        problem = _count_problem(spec, counts)
        if problem:
            raise _SourceError(problem)
        _install(stage, dest, work)
        return fetched, counts
    finally:
        _rmtree(work)


def _process_dataset(spec: _DatasetSpec, ctx: _Context, force: bool) -> _Result:
    """Make one dataset available: skip it when complete, else try its sources in order."""
    dest = ctx.root / spec.name
    counts = _split_counts(dest, spec.splits)
    if not force and _counts_complete(spec, counts, ctx.report):
        ctx.report.info(f"{spec.name}: already complete ({_format_counts(counts)}) in {dest}")
        return _Result(spec.name, True, "already complete", dest, counts, "-")

    if force and os.path.lexists(dest):
        state = "re-downloading (--force; the current folder is kept until the new copy is ready)"
    elif any(counts.values()):
        state = f"incomplete ({_format_counts(counts)}), downloading"
    else:
        state = "downloading"
    ctx.report.info(f"{spec.name}: {state} into {dest}")
    if spec.name not in _LICENCES_SHOWN:
        _LICENCES_SHOWN.add(spec.name)
        ctx.report.info(f"  licence: {LICENCE_NOTES[spec.name]}")

    plan = _plan_sources(spec, ctx.downloads)
    errors: list[str] = []
    for number, source in enumerate(plan, 1):
        ctx.report.info(f"  [{number}/{len(plan)}] {source.label}")
        try:
            fetched, counts = _attempt(spec, source, ctx, dest)
        except Exception as exc:  # any failure moves on to the next source
            reason = str(exc) if isinstance(exc, _SourceError) else f"{type(exc).__name__}: {exc}"
            reason = " ".join(reason.split())  # always a single line
            ctx.report.info(f"        failed: {reason}")
            errors.append(f"{source.key}: {reason}")
            continue

        warnings = list(fetched.warnings)
        note = _count_note(spec, counts)
        if note:
            warnings.append(note)
        for warning in warnings:
            ctx.report.warn(f"{spec.name}: {warning}")
        ctx.report.info(f"        ok: {_format_counts(counts)}; {fetched.verification}")
        entry: dict[str, Any] = {
            "source": source.key,
            "description": source.label,
            "urls": fetched.urls,
            "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "file_count": counts,
            "verification": fetched.verification,
            "licence": LICENCE_NOTES[spec.name],
        }
        if fetched.note:
            entry["note"] = fetched.note
        if warnings:
            entry["warnings"] = warnings
        if fetched.deviations:
            entry["manifest_deviations"] = fetched.deviations
        if errors:
            entry["failed_sources"] = errors
        _write_provenance(ctx, spec.name, entry)
        return _Result(spec.name, True, "downloaded", dest, counts, source.key)

    ctx.report.warn(f"{spec.name}: all {len(plan)} sources failed")
    return _Result(
        spec.name, False, "FAILED", dest, source=f"{len(plan)} sources failed", errors=errors
    )


def _counts_complete(spec: _DatasetSpec, counts: dict[str, int], report: _Reporter) -> bool:
    """Completeness rule of is_complete(); warns about accepted-but-unexpected counts."""
    if _count_problem(spec, counts):
        return False
    note = _count_note(spec, counts)
    if note:
        report.warn(f"{spec.name}: {note}")
    return True


def _download_datasets(
    names: Sequence[str], root: Path, force: bool, timeout: float, quiet: bool
) -> list[_Result]:
    """Process every dataset in turn; a failure does not stop the others."""
    if not (timeout > 0 and math.isfinite(timeout)):
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")
    ctx = _Context(root, float(timeout), _Reporter(quiet))
    try:
        return [_process_dataset(_SPECS[name], ctx, force) for name in names]
    finally:
        _remove_if_empty(ctx.downloads)


def _manual_instructions(spec: _DatasetSpec, root: Path) -> str:
    """How to download a dataset by hand: URL(s), drop-in path(s), checksums, unpack hint."""
    downloads = root / DOWNLOADS_DIRNAME
    archives = next(s for s in spec.sources if isinstance(s, _ArchiveSource)).archives
    what = "this file" if len(archives) == 1 else f"these {len(archives)} files"
    lines = [
        f"  Manual download: fetch {what} (browser or another network), save as shown, then re-run:"
    ]
    for archive in archives:
        checksum = f"  (sha256 {archive.sha256})" if archive.sha256 else ""
        lines += [f"    {archive.url}", f"      -> {downloads / archive.filename}{checksum}"]
    lines.append("  " + spec.unpack_hint.replace("{root}", str(root)))
    return "\n".join(lines)


def _unavailable_message(failed: Sequence[_Result], root: Path) -> str:
    """The DatasetUnavailableError message: reasons per source plus manual instructions."""
    lines = [f"Could not download {', '.join(r.name for r in failed)}: every source failed."]
    for result in failed:
        lines.append(f"{result.name}:")
        lines += [f"  - {error}" for error in result.errors]
        lines.append(_manual_instructions(_SPECS[result.name], root))
        if os.path.lexists(result.path):
            lines.append(f"  The existing folder {result.path} was left unchanged.")
    return "\n".join(lines)


def _verify(spec: _DatasetSpec, root: Path, report: _Reporter) -> _Result:
    """Recount a dataset and re-hash every file that has a manifest entry."""
    folder = root / spec.name
    entry = _read_provenance(root).get(spec.name)
    source = str(entry.get("source", "-")) if isinstance(entry, dict) else "-"
    if not folder.is_dir():
        report.info(f"{spec.name}: FAILED - not downloaded ({folder} does not exist)")
        return _Result(spec.name, False, "FAILED", folder, source=source)

    counts = _split_counts(folder, spec.splits)
    problems = []
    if not _counts_complete(spec, counts, report):
        problems.append(_count_problem(spec, counts) or "incomplete")
    manifest = _manifest_files(spec.name)
    if manifest:
        checked, deviations = _compare_with_manifest(spec.name, folder)
        known = _known_deviations(root, spec.name)
        accepted = sum(1 for rel, sha in deviations.items() if known.get(rel) == sha)
        for rel, actual in sorted(deviations.items()):
            if rel not in known:
                problems.append(_mismatch(rel, manifest[rel][0], actual))
            elif known[rel] != actual:  # an accepted deviation that has changed since download
                problems.append(
                    _mismatch(rel, known[rel], actual) + "; expected = hash in PROVENANCE.json"
                )
        hashes = f"sha256 of {checked} files checked against dataset_manifest.json"
        if accepted:
            hashes += f" ({accepted} known deviations recorded at download time accepted)"
    elif _load_manifest() is None:
        hashes = "sha256 not checked (dataset_manifest.json missing)"
    else:
        hashes = "no per-file checksums for this dataset (archives were verified when downloaded)"

    if problems:
        report.info(f"{spec.name}: FAILED ({_format_counts(counts)}; {hashes})")
        for problem in problems:
            report.info(f"  - {problem}")
        return _Result(spec.name, False, "FAILED", folder, counts, source)
    report.info(f"{spec.name}: OK ({_format_counts(counts)}; {hashes})")
    return _Result(spec.name, True, "verified", folder, counts, source)


# ----------------------------------------------------------------------------- public API


def default_root() -> Path:
    """The default data folder: ``data/`` next to this script."""
    return Path(__file__).resolve().parent / "data"


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    return default_root() if root is None else Path(root).expanduser().resolve()


def dataset_dir(name: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Folder of a dataset (``<root>/<NAME>``); name is case-insensitive."""
    return _resolve_root(root) / _spec(name).name


def is_complete(name: str, root: str | os.PathLike[str] | None = None) -> bool:
    """True when every split folder holds exactly the expected number of images.

    ICDAR2003 is the exception: any non-zero count in both splits is complete, but a count
    that differs from the expected one prints a warning.
    """
    spec = _spec(name)
    counts = _split_counts(dataset_dir(spec.name, root), spec.splits)
    return _counts_complete(spec, counts, _Reporter())


def ensure_datasets(
    names: str | Iterable[str] = ("BSD300", "SET14"),
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
    timeout: float = 30.0,
    quiet: bool = False,
) -> dict[str, Path]:
    """Download the named datasets unless they are already complete; return name -> folder.

    Names are case-insensitive ("all" selects every dataset). ``root`` defaults to
    ``default_root()``; relative paths are taken from the current directory. Complete datasets
    are skipped without network access unless ``force`` is true, which downloads a fresh copy
    and replaces the old folder only after success. ``timeout`` is the per-request network
    timeout in seconds. ``quiet`` hides progress output (warnings are still printed).

    Raises DatasetUnavailableError after processing the other datasets if every source failed
    for some dataset. The error message explains the manual download.
    """
    root_path = _resolve_root(root)
    results = _download_datasets(_canonical_names(names), root_path, force, timeout, quiet)
    available = {r.name: r.path for r in results if r.ok}
    failed = [r for r in results if not r.ok]
    if failed:
        raise DatasetUnavailableError(
            _unavailable_message(failed, root_path), {r.name: r.errors for r in failed}, available
        )
    return available


def verify(name: str, root: str | os.PathLike[str] | None = None) -> bool:
    """Recount a dataset and re-hash its files against the manifest where an entry exists.

    Manifest deviations that were accepted at download time (official archives whose bytes
    may differ from the mirror; recorded in PROVENANCE.json) do not count as failures.
    Prints one line per problem and returns True when everything checks out.
    """
    return _verify(_spec(name), _resolve_root(root), _Reporter()).ok


# ----------------------------------------------------------------------------- command line


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (value > 0 and math.isfinite(value)):
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the BSD300, SET14 and ICDAR2003 datasets for the PBL notebook "
        "(Python standard library only).",
        epilog="Exit code: 0 when every requested dataset is available, 1 otherwise.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help="bsd300, set14, icdar2003 or all (default: all; case-insensitive)",
    )
    parser.add_argument("--root", metavar="DIR", help=f"data folder (default: {default_root()})")
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even if complete; the old folder is replaced only after success",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="recount and re-hash the files; no downloads"
    )
    parser.add_argument(
        "--list-sources", action="store_true", help="print the ordered sources per dataset and exit"
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        metavar="SEC",
        help="network timeout per request in seconds (default: 30)",
    )
    # For testing only: SLOT=URL[#sha256=HEX], see the module docstring.
    parser.add_argument("--source-override", action="append", default=[], help=argparse.SUPPRESS)
    return parser


def _print_sources(names: Sequence[str], root: Path) -> None:
    """--list-sources: the sources of each dataset in the order they are tried."""
    downloads = root / DOWNLOADS_DIRNAME
    for name in names:
        spec = _SPECS[name]
        sources = [_with_overrides(source) for source in spec.sources]
        drop_ins = next(s for s in sources if isinstance(s, _ArchiveSource)).archives
        print(f"{name} (expected {_format_counts(spec.splits)})")
        paths = " + ".join(str(downloads / a.filename) for a in drop_ins)
        print(f"  0. drop-in archive, used first when present: {paths}")
        for number, source in enumerate(sources, 1):
            print(f"  {number}. {source.label}")
            for slot, url, check in source.slots():
                print(f"       {slot:<22} {url}\n       {'':<22} {check}")


def _print_summary(results: Sequence[_Result], root: Path) -> None:
    """The table printed at the end of a command-line run."""
    rows = [("dataset", "status", "images", "source")]
    rows += [(r.name, r.status, _format_counts(r.counts) or "-", r.source or "-") for r in results]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    print(f"\nSummary (root: {root})")
    for row in rows:
        print(
            "  "
            + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        names = _canonical_names(args.datasets)
        _set_source_overrides(args.source_override)
    except ValueError as exc:
        parser.error(str(exc))
    root = _resolve_root(args.root)

    if args.list_sources:
        _print_sources(names, root)
        return 0
    if args.verify_only:
        report = _Reporter()
        results = [_verify(_SPECS[name], root, report) for name in names]
        _print_summary(results, root)
        return 0 if all(r.ok for r in results) else 1

    try:
        results = _download_datasets(names, root, args.force, args.timeout, quiet=False)
    except KeyboardInterrupt:
        print("\nInterrupted; temporary files were removed.", file=sys.stderr)
        return 130
    _print_summary(results, root)
    failed = [r for r in results if not r.ok]
    if failed:
        _write(sys.stderr, "\n" + _unavailable_message(failed, root) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
