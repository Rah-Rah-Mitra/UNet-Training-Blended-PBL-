#!/usr/bin/env python3
"""Download the image datasets for the PBL super-resolution notebook (standard library only).

The script needs nothing but Python >= 3.10, so it can run before any dependency is
installed (no ``requests``, no PIL). By default it fills ``data/`` next to this file, i.e.
next to ``SimplifiedUNetSR.ipynb``::

    data/
      BSD300/train/*.jpg    200 images, BSDS300 train split (original names, e.g. 100075.jpg)
      BSD300/test/*.jpg     100 images, BSDS300 test split
      SET14/train/*.png      11 images: the 14 Set14 images (baboon.png, ... zebra.png) except
      SET14/test/*.png        3 images: comic.png, monarch.png and zebra.png (the notebook's split)
      ICDAR2003/train/*     258 images, ICDAR 2003 Robust Reading "SceneTrialTrain"
      ICDAR2003/test/*      251 images, "SceneTrialTest" (the paper reports 249)
      PROVENANCE.json       per dataset: source, URLs, UTC time, counts, verification, licence

Image files are stored byte for byte as downloaded; nothing is decoded or re-encoded. Only the
split folders (train/, test/) are written; anything else inside a dataset folder is left alone.

Command line::

    python download_datasets.py                           # every dataset
    python download_datasets.py --datasets bsd300 set14   # a subset (case-insensitive)
    python download_datasets.py --root "D:/My Data"       # another data folder
    python download_datasets.py --force                   # re-download; old copy kept until success
    python download_datasets.py --verify-only             # recount and re-hash; no downloads
    python download_datasets.py --list-sources            # the ordered sources of each dataset
    python download_datasets.py --timeout 60              # network timeout in seconds (default 30;
                                                          # see "How it works")

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
* The timeout (``--timeout``, default 30 s) applies to connecting and to every wait for data:
  an attempt fails when the server sends nothing for that long, and also when a download
  makes less than 1 kB of progress in that time (a stalled or trickling connection). It is
  not a limit on the total download time, so large files work on slow connections.
* Each attempt works in its own temporary folder in ``<root>/.downloads/``: downloads are
  streamed to temporary files there and moved into place when complete, and archives are
  extracted there. The split folders of a dataset are created or replaced only after the new
  copy has been fully verified, so a failed attempt leaves nothing behind. Leftovers of a run
  that was killed (e.g. a kernel restart) are deleted by the next run; a lock file per
  temporary folder keeps that from touching a download that is still running.
* Ctrl-C (or interrupting the Jupyter kernel) stops a run at once; no new requests or retries
  start after it.
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
import os
import queue
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
    "SET14": {"train": 11, "test": 3},  # test = comic, monarch, zebra (see _SET14_TEST)
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
STALL_BYTES = 1024  # a download must make this much progress within every timeout period
MAX_TIMEOUT = 86400.0  # 1 day; far larger values overflow the socket layer
MIRROR_WORKERS = 8  # parallel requests for the one-request-per-file GitHub mirrors
WORKER_GRACE = 1.0  # seconds to wait for busy download threads after a failure or Ctrl-C

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
# The notebook's SET14 test split (the paper's bicubic numbers are reproduced only with these
# three images); the other 11 images are the training split.
_SET14_TEST = frozenset({"comic", "monarch", "zebra"})
_SET14_ALIASES = {"lena": "lenna", "peppers": "pepper"}  # spellings used by some Set14 copies

_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")
_SAFE_RELATIVE = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-][A-Za-z0-9._-]*")  # "<split>/<file>"
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_TAR_DATA_FILTER = hasattr(tarfile, "data_filter")  # Python >= 3.12 and the security backports
# What this script creates in <root>/.downloads/: a lock file and a work folder per attempt
# ("<NAME>-<random>.lock" / ".tmp"); older versions also left "download-<random>.part" there.
_TEMPORARY_NAME = re.compile(
    r"(?:BSD300|SET14|ICDAR2003)-[a-z0-9_]+\.(?:tmp|lock)|download-[a-z0-9_]+\.part"
)

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


class _Cancelled(Exception):
    """A download thread stopped because the run failed elsewhere or was interrupted."""


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


def _print(text: str = "") -> None:
    """print() for the command line that cannot fail on characters the console lacks."""
    _write(sys.stdout, text + "\n")


def _safe_console() -> None:
    """Let stdout/stderr replace unencodable characters instead of raising.

    A path such as ``C:/Users/Иван/data`` cannot be printed to a cp1252 pipe or file, which
    would otherwise end a successful run with a UnicodeEncodeError (argparse's --help too).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and getattr(stream, "errors", "strict") == "strict":
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


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

    ``info`` lines and progress are suppressed when quiet; warnings always go to stderr. On
    terminals and in Jupyter the progress line redraws itself with ``\\r``. Elsewhere (a pipe,
    a file, a CI log) progress is logged as ordinary lines: one per 10 % step at most every
    2 s, or every 10 s when the total size is unknown, so captured logs stay readable.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._out: TextIO = sys.stdout
        self._live = not quiet and _is_interactive(self._out)
        self._lock = threading.Lock()
        self._progress_width = 0  # length of the progress line on screen (0: none)
        self._last_draw = 0.0  # live mode: time of the last redraw
        self._last_logged = 0.0  # log mode: time of the last progress line (or of the start)
        self._last_step = 0  # log mode: last 10 % step that was logged

    def info(self, message: str) -> None:
        """Print a normal message unless quiet."""
        if not self.quiet:
            self._line(self._out, message)

    def warn(self, message: str) -> None:
        """Print a warning to stderr, even when quiet."""
        self._line(sys.stderr, f"WARNING: {message}")

    def start_progress(self) -> None:
        """A new transfer starts: restart the spacing of logged progress lines."""
        self._last_logged = time.monotonic()
        self._last_step = 0

    def progress(self, text: str, fraction: float | None = None) -> None:
        """Show progress: redraw the live line (at most 5 times per second) or log a line.

        ``fraction`` is the share done (0..1), or None when the total is unknown.
        """
        if self.quiet:
            return
        now = time.monotonic()
        if self._live:
            if now - self._last_draw < 0.2:
                return
            with self._lock:
                self._last_draw = now
                _write(self._out, "\r" + text.ljust(self._progress_width))
                self._progress_width = len(text)
            return
        if fraction is None:
            if now - self._last_logged < 10.0:
                return
        else:
            step = int(fraction * 10)
            if step <= self._last_step or step >= 10 or now - self._last_logged < 2.0:
                return
            self._last_step = step
        self._last_logged = now
        self._line(self._out, text)

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
    # differences from dataset_manifest.json that were accepted (official archives only):
    # "split/file" -> sha256 on disk, or None for a manifest file the archive did not contain
    deviations: dict[str, str | None] = field(default_factory=dict)
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
        manifest = _compare_with_manifest(spec, stage)
        deviations = manifest.deviations() if manifest else {}
        if manifest:
            checks.append(f"{manifest.matched}/{manifest.total} images match dataset_manifest.json")
        if manifest and deviations:
            warnings.append(
                f"the images differ from dataset_manifest.json ({manifest.describe()}); kept, "
                "because this archive's bytes may legitimately differ from the mirror "
                "(recorded in PROVENANCE.json)"
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


def _image_names(folder: Path) -> list[str]:
    """Sorted names of the image files directly inside folder ([] if it does not exist)."""
    try:
        with os.scandir(folder) as entries:
            return sorted(e.name for e in entries if _is_image_name(e.name) and e.is_file())
    except (FileNotFoundError, NotADirectoryError):
        return []


def _count_images(folder: Path) -> int:
    """Number of image files directly inside folder (0 if it does not exist)."""
    return len(_image_names(folder))


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


def _some(items: Sequence[str], limit: int = 5) -> str:
    """Comma-separated items, cut after ``limit`` with "... (+N more)", for long lists."""
    shown = ", ".join(items[:limit])
    return f"{shown}, ... (+{len(items) - limit} more)" if len(items) > limit else shown


def _set14_canonical(file_name: str) -> str | None:
    """Canonical Set14 name ("baboon" ... "zebra") of an image file name, or None if unknown.

    Accepts the canonical names in any case ("Baboon.PNG"), with a suffix such as "_HR" or
    "-GT", the spellings "lena" and "peppers", and the numbered names of SelfExSR-style copies
    ("img_001_SRF_2_HR.png" is baboon ... "img_014_SRF_2_HR.png" is zebra).
    """
    stem = os.path.splitext(file_name)[0].lower()
    numbered = re.fullmatch(r"img_?0*(\d{1,2})(?:[_-].*)?", stem)
    if numbered:
        number = int(numbered.group(1))
        return _SET14_NAMES[number - 1] if 1 <= number <= len(_SET14_NAMES) else None
    stem = re.sub(r"[_-](?:hr|gt|original)$", "", stem)
    stem = _SET14_ALIASES.get(stem, stem)
    return stem if stem in _SET14_NAMES else None


def _set14_split(canonical: str) -> str:
    """The SET14 split of a canonical image name: comic, monarch and zebra are the test set."""
    return "test" if canonical in _SET14_TEST else "train"


def _layout_path(name: str, rel: str) -> str:
    """Where a manifest path ("<split>/<file>") lives in this script's layout.

    SET14 images are sorted into train/ and test/ by image, whatever split the manifest key
    names: its keys predate the 11/3 split and are all "test/<name>.png".
    """
    if name == "SET14":
        file_name = rel.rpartition("/")[2]
        canonical = _set14_canonical(file_name)
        if canonical:
            return f"{_set14_split(canonical)}/{file_name}"
    return rel


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rmtree(path: Path, *, quiet: bool = False) -> bool:
    """Delete a folder tree (best effort), making read-only files writable first (Windows).

    Returns False (after a warning unless quiet) when something could not be deleted.
    """

    def make_writable_and_retry(
        function: Callable[[str], object], target: str, _error: object
    ) -> None:
        os.chmod(target, stat.S_IWRITE)
        function(target)

    if not os.path.lexists(path):
        return True
    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=make_writable_and_retry)
        else:
            shutil.rmtree(path, onerror=make_writable_and_retry)
    except OSError as exc:
        if not quiet:
            _warn(f"could not remove the temporary folder {path}: {exc}")
        return False
    return True


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

    The keys are paths in this script's layout (see _layout_path). Returns None when the
    manifest is missing or has no files for this dataset. Entries that are not images of an
    expected split (such as the BSDS ``iids_*.txt`` lists) are ignored.
    """
    manifest = _load_manifest()
    section = manifest.get(name) if manifest else None
    files = section.get("files") if isinstance(section, dict) else None
    if not isinstance(files, dict):
        return None
    entries: dict[str, tuple[str, str | None]] = {}
    for rel, value in files.items():
        if not _SAFE_RELATIVE.fullmatch(rel):
            continue
        path = _layout_path(name, rel)
        if path.split("/")[0] not in EXPECTED_COUNTS[name]:
            continue
        if isinstance(value, dict):  # {"source": "<file on the mirror>", "sha256": "..."}
            sha256, source = value.get("sha256"), value.get("source")
        else:  # plain "<sha256>"
            sha256, source = value, None
        valid_source = source is None or (isinstance(source, str) and _SAFE_NAME.fullmatch(source))
        if not (isinstance(sha256, str) and _SHA256_HEX.fullmatch(sha256) and valid_source):
            _warn(f"ignoring the malformed {MANIFEST_PATH.name} entry {name}: {rel}")
            continue
        entries[path] = (sha256.lower(), source)
    return entries or None


@dataclass
class _ManifestCheck:
    """A dataset folder compared with the dataset's entries in dataset_manifest.json."""

    total: int  # number of manifest entries
    mismatched: dict[str, str] = field(default_factory=dict)  # "split/file" -> sha256 on disk
    missing: list[str] = field(default_factory=list)  # listed files that are not on disk
    unlisted: dict[str, str] = field(default_factory=dict)  # images on disk without an entry

    @property
    def checked(self) -> int:
        """Listed files that are on disk (each was hashed)."""
        return self.total - len(self.missing)

    @property
    def matched(self) -> int:
        """Listed files that are on disk with the listed sha256."""
        return self.checked - len(self.mismatched)

    def deviations(self) -> dict[str, str | None]:
        """Every difference: "split/file" -> sha256 on disk, or None for a missing file."""
        found: dict[str, str | None] = {**self.mismatched, **self.unlisted}
        found.update(dict.fromkeys(self.missing))
        return dict(sorted(found.items()))

    def describe(self) -> str:
        """Short summary of the differences, e.g. "3 of 300 listed files missing"."""
        parts = []
        if self.mismatched:
            parts.append(f"{len(self.mismatched)} of {self.checked} with a different sha256")
        if self.missing:
            parts.append(f"{len(self.missing)} of {self.total} listed files missing")
        if self.unlisted:
            parts.append(f"{len(self.unlisted)} images not listed")
        return ", ".join(parts) or "no differences"


def _compare_with_manifest(spec: _DatasetSpec, folder: Path) -> _ManifestCheck | None:
    """Compare a dataset folder with its manifest entries (None: the manifest has none).

    Every listed file is hashed. A listed file that is not on disk counts as missing, and an
    image in a split folder that has no entry counts as unlisted (names compared ignoring case,
    as on Windows and macOS file systems).
    """
    entries = _manifest_files(spec.name)
    if not entries:
        return None
    check = _ManifestCheck(total=len(entries))
    for rel, (expected, _source) in sorted(entries.items()):
        path = folder / rel
        if not path.is_file():
            check.missing.append(rel)
            continue
        actual = _sha256_file(path)
        if actual != expected:
            check.mismatched[rel] = actual
    listed = {rel.lower() for rel in entries}
    for split in spec.splits:
        for file_name in _image_names(folder / split):
            rel = f"{split}/{file_name}"
            if rel.lower() not in listed:
                check.unlisted[rel] = _sha256_file(folder / rel)
    return check


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


def _known_deviations(root: Path, name: str) -> dict[str, str | None]:
    """Manifest deviations accepted at download time (recorded in PROVENANCE.json).

    "split/file" -> sha256 the file had, or None for a listed file the download did not have.
    """
    entry = _read_provenance(root).get(name)
    deviations = entry.get("manifest_deviations") if isinstance(entry, dict) else None
    if not isinstance(deviations, dict):
        return {}
    return {
        k: v
        for k, v in deviations.items()
        if isinstance(k, str) and (v is None or isinstance(v, str))
    }


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


def _download_once(
    url: str,
    dest: Path,
    ctx: _Context,
    progress_label: str | None,
    cancel: threading.Event | None,
) -> _Download:
    """Stream url into a temporary file next to dest, then os.replace it to dest.

    dest is always inside the attempt's work folder in <root>/.downloads/, so the temporary
    file is too, and it disappears with that folder whatever happens. The request fails when
    connecting or a read takes longer than ctx.timeout, and when less than STALL_BYTES arrive
    within ctx.timeout (a stalled or trickling transfer). A set ``cancel`` stops it.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=dest.parent)
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
            read = getattr(response, "read1", response.read)  # read1: returns once data arrives
            mark_time, mark_size = time.monotonic(), 0  # last time STALL_BYTES had arrived
            if progress_label:
                ctx.report.start_progress()
            while chunk := read(CHUNK_SIZE):
                if cancel is not None and cancel.is_set():
                    raise _Cancelled()
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                now = time.monotonic()
                if size - mark_size >= STALL_BYTES:
                    mark_time, mark_size = now, size
                elif now - mark_time > ctx.timeout:
                    raise _SourceError(
                        f"stalled: less than {_size(STALL_BYTES)} received in {ctx.timeout:g}s",
                        retryable=True,
                    )
                if progress_label:
                    shown = (
                        f"{_size(size)} / {_size(total)} ({100 * size // total}%)"
                        if total
                        else _size(size)
                    )
                    fraction = size / total if total else None
                    ctx.report.progress(f"        {progress_label}: {shown}", fraction)
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


def _download(
    url: str,
    dest: Path,
    ctx: _Context,
    progress_label: str | None = None,
    cancel: threading.Event | None = None,
) -> _Download:
    """Download url to dest, retrying transient failures with exponential backoff.

    Raises _SourceError with a one-line reason when the URL cannot be fetched, and _Cancelled
    as soon as ``cancel`` is set: no new attempt or backoff wait starts after that.
    """
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        if cancel is not None and cancel.is_set():
            raise _Cancelled()
        try:
            return _download_once(url, dest, ctx, progress_label, cancel)
        except _Cancelled:
            raise
        except Exception as exc:  # classified below; KeyboardInterrupt is not caught
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()  # an error response still holds its connection
            reason, retryable = _explain_error(exc, url, ctx.timeout)
            if not retryable or attempt == HTTP_ATTEMPTS:
                raise _SourceError(
                    reason + (f" (after {attempt} attempts)" if attempt > 1 else "")
                ) from exc
            if cancel is not None and cancel.is_set():
                raise _Cancelled() from exc
            delay = RETRY_BACKOFF[attempt - 1]
            ctx.report.info(
                f"        {reason}; attempt {attempt + 1}/{HTTP_ATTEMPTS} in {delay:g}s: {url}"
            )
            if cancel is None:
                time.sleep(delay)
            elif cancel.wait(delay):
                raise _Cancelled() from exc
    raise AssertionError("unreachable")


def _download_files(files: Sequence[_MirrorFile], stage: Path, ctx: _Context) -> None:
    """Fetch mirror files in parallel into stage, verifying each checksum (mismatch = failure).

    The download threads are daemon threads. When a file fails, or on Ctrl-C (or a Jupyter
    kernel interrupt), the other threads stop at their next step - no new request, retry or
    backoff wait starts - and this function raises after at most WORKER_GRACE seconds, even
    when a request is stuck in the network. Its temporary file is inside the work folder, which
    the caller deletes.
    """
    cancel = threading.Event()
    pending: queue.SimpleQueue[_MirrorFile] = queue.SimpleQueue()
    results: queue.SimpleQueue[tuple[int, BaseException | None]] = queue.SimpleQueue()
    for item in files:
        pending.put(item)

    def fetch_one(item: _MirrorFile) -> int:
        try:
            result = _download(item.url, stage / item.dest, ctx, cancel=cancel)
        except _SourceError as exc:
            raise _SourceError(f"{item.dest}: {exc}") from exc
        if item.sha256 and result.sha256 != item.sha256:
            raise _SourceError(_mismatch(item.dest, item.sha256, result.sha256))
        return result.size

    def worker() -> None:
        while not cancel.is_set():
            try:
                item = pending.get_nowait()
            except queue.Empty:
                return
            try:
                results.put((fetch_one(item), None))
            except BaseException as exc:  # handed over to the main thread
                results.put((0, exc))
                return

    workers = [
        threading.Thread(target=worker, name=f"mirror-download-{number}", daemon=True)
        for number in range(min(MIRROR_WORKERS, len(files)))
    ]
    started, done, total = time.monotonic(), 0, 0
    ctx.report.start_progress()
    try:
        for thread in workers:
            thread.start()
        while done < len(files):
            try:  # a timeout keeps Ctrl-C working on Windows, where a plain get() blocks it
                size, error = results.get(timeout=0.25)
            except queue.Empty:
                if results.empty() and not any(thread.is_alive() for thread in workers):
                    raise _SourceError("the download threads stopped unexpectedly") from None
                continue
            if error is not None:
                raise error
            total += size
            done += 1
            fraction = done / len(files)
            ctx.report.progress(f"        {done}/{len(files)} files, {_size(total)}", fraction)
    finally:
        cancel.set()
        deadline = time.monotonic() + WORKER_GRACE
        for thread in workers:
            if thread.ident is not None:  # started
                thread.join(max(0.0, deadline - time.monotonic()))
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


def _layout_set14(extracted: Path, _split: str) -> Iterator[tuple[str, Path, str]]:
    """Set14 tarball (unknown inner layout): every image, under its canonical name.

    Each image goes to train/ or test/ (comic, monarch, zebra) by its name, e.g.
    ``Set14/HR/Zebra.PNG`` becomes ``test/zebra.png``. An image whose name is not one of the
    14 Set14 images fails the source, since its split cannot be known.
    """
    for path in _image_files(extracted):
        canonical = _set14_canonical(path.name)
        if canonical is None:
            raise _SourceError(
                f"unknown image {path.name!r} in the archive: expected the 14 Set14 images "
                "(baboon.png ... zebra.png or img_001 ... img_014)"
            )
        yield _set14_split(canonical), path, canonical + path.suffix.lower()


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
    """Set14 files on the mirror, saved under canonical names in train/ or test/.

    Names and hashes come from the manifest, else from the built-in list (img_001 = baboon ...).
    """
    entries = _manifest_files("SET14")
    if entries and all(source for _, source in entries.values()):
        return [
            _MirrorFile(rel, f"{base}/{source}", sha)
            for rel, (sha, source) in sorted(entries.items())
        ]
    return [
        _MirrorFile(
            f"{_set14_split(name)}/{name}.png", f"{base}/img_{number:03d}_SRF_2_HR.png", None
        )
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
                (_Archive("set14-hf", _SET14_TARBALL, "Set14_HR.tar.gz"),),
                _layout_set14,
            ),
        ),
        unpack_hint=(
            "(Or unpack the 14 images it contains yourself: comic, monarch and zebra into "
            "{root}/SET14/test/, the other 11 into {root}/SET14/train/.)"
        ),
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


# ------------------------------------------------------------------- temporary folders and locks


def _try_lock(fd: int) -> bool | None:
    """Try to lock an open file exclusively, without waiting.

    True: locked; the lock ends when fd is closed or the process dies (even by SIGKILL).
    False: another process holds it. None: this file system cannot lock files.
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:  # POSIX: locked by another process
        return False
    except OSError as exc:
        if sys.platform == "win32" and exc.errno in (errno.EACCES, errno.EDEADLK):
            return False  # msvcrt: the byte is locked by another process
        return None
    return True


def _lock_is_free(path: Path) -> bool:
    """True when no running process holds the lock file at path (its run ended or died)."""
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return True
    except OSError:  # cannot tell, e.g. no permission: treat it as in use
        return False
    try:
        return _try_lock(fd) is True  # None (no locking support): treat it as in use
    finally:
        os.close(fd)  # gives the lock back at once


@contextlib.contextmanager
def _work_folder(downloads: Path, name: str) -> Iterator[Path]:
    """A new work folder ``<downloads>/<name>-<random>.tmp`` for one attempt, deleted after it.

    The lock file ``<name>-<random>.lock`` next to it stays locked while the folder is in use,
    so the clean-up in another run (_remove_stale_temporaries) never deletes it too early.
    """
    downloads.mkdir(parents=True, exist_ok=True)
    fd, lock_name = tempfile.mkstemp(prefix=f"{name}-", suffix=".lock", dir=downloads)
    lock = Path(lock_name)
    work = lock.with_suffix(".tmp")
    try:
        _try_lock(fd)
        if os.path.lexists(work):  # an orphan with the same name but no lock file: stale
            _rmtree(work, quiet=True)
        work.mkdir()
        yield work
    finally:
        _rmtree(work)
        os.close(fd)
        with contextlib.suppress(OSError):
            lock.unlink()


def _remove_stale_temporaries(downloads: Path, report: _Reporter) -> None:
    """Delete what killed runs (e.g. a kernel restart) left in ``<root>/.downloads/``.

    That is: work folders whose lock is no longer held, lock files without a folder, and the
    ``download-*.part`` files of older versions. Work folders of runs still in progress are
    kept, and drop-in archives or anything else in the folder are never touched.
    """
    try:
        with os.scandir(downloads) as scan:
            entries = {entry.name: entry.is_dir(follow_symlinks=False) for entry in scan}
    except OSError:  # no .downloads folder (the usual case) or unreadable
        return
    removed = 0
    for name, is_dir in sorted(entries.items()):
        if not _TEMPORARY_NAME.fullmatch(name):
            continue
        path = downloads / name
        stem, _, kind = name.rpartition(".")
        if kind == "tmp" and is_dir:
            lock = downloads / f"{stem}.lock"
            if lock.name in entries and not _lock_is_free(lock):
                continue  # in use by a running download
            if _rmtree(path, quiet=True):
                removed += 1
            with contextlib.suppress(OSError):
                lock.unlink()
        elif kind == "lock" and not is_dir and f"{stem}.tmp" not in entries:
            # A lock file is created a moment before it is locked and before its folder exists:
            # only lock files older than a minute can be orphans.
            with contextlib.suppress(OSError):
                if time.time() - path.stat().st_mtime > 60 and _lock_is_free(path):
                    path.unlink()
        elif kind == "part" and not is_dir:
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
    if removed:
        report.info(f"removed {removed} leftover(s) of an interrupted earlier run from {downloads}")


# -------------------------------------------------------------------------- downloading a dataset


def _move_aside(target: Path, into: Path, beside: Path) -> Path:
    """Rename target to ``into``, or to ``beside`` (next to target) across drives; new place."""
    try:
        os.replace(target, into)
        return into
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    os.replace(target, beside)
    return beside


def _move_in(source: Path, target: Path, beside: Path) -> None:
    """Rename source to target; across drives copy it to ``beside`` first, then rename."""
    try:
        os.replace(source, target)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    try:
        shutil.copytree(source, beside, symlinks=True)
        os.replace(beside, target)
    except BaseException:
        _rmtree(beside, quiet=True)
        raise


def _install(stage: Path, dest: Path, work: Path, splits: Iterable[str]) -> None:
    """Move the verified split folders from stage into dest; on failure undo everything.

    Only the split folders are replaced: any other file or folder in dest is kept. An old split
    folder is moved aside (into the work folder) before its new copy moves in, and deleted only
    after all of them are in place. A dest that is a link to a folder elsewhere is replaced by
    a real folder; the folder it points to is never modified.
    """
    tag = work.name.removesuffix(".tmp")  # unique per attempt
    if os.path.islink(dest) or not os.path.lexists(dest):
        moves = [(stage, dest)]
    else:
        moves = [(stage / split, dest / split) for split in splits]
    (work / "previous").mkdir(exist_ok=True)
    moved_aside: list[tuple[Path, Path]] = []  # (where an old copy is now, where it was)
    moved_in: list[Path] = []
    try:
        for new, target in moves:
            if os.path.lexists(target):
                beside = target.with_name(f".{target.name}.old-{tag}")
                moved_aside.append(
                    (_move_aside(target, work / "previous" / target.name, beside), target)
                )
            _move_in(new, target, target.with_name(f".{target.name}.new-{tag}"))
            moved_in.append(target)
    except BaseException:
        for target in reversed(moved_in):
            _rmtree(target, quiet=True)
        for aside, target in reversed(moved_aside):
            try:
                os.replace(aside, target)
            except OSError as exc:
                _warn(f"could not move {aside} back to {target}: {exc}")
        raise
    for aside, _target in moved_aside:
        if not aside.is_relative_to(work):  # was moved beside its original place
            if os.path.islink(aside):
                os.unlink(aside)
            else:
                _rmtree(aside)


def _attempt(
    spec: _DatasetSpec, source: _Source, ctx: _Context, dest: Path
) -> tuple[_Fetched, dict[str, int]]:
    """Run one source in a fresh work folder and install the result if it is complete.

    The work folder (downloads, extracted files, staged images, replaced split folders) is
    always deleted.
    """
    with _work_folder(ctx.downloads, spec.name) as work:
        stage = work / "stage"
        for split in spec.splits:
            (stage / split).mkdir(parents=True)
        fetched = source.fetch(spec, ctx, work)
        counts = _split_counts(stage, spec.splits)
        problem = _count_problem(spec, counts)
        if problem:
            raise _SourceError(problem)
        _install(stage, dest, work, spec.splits)
        return fetched, counts


def _process_dataset(spec: _DatasetSpec, ctx: _Context, force: bool) -> _Result:
    """Make one dataset available: skip it when complete, else try its sources in order."""
    dest = ctx.root / spec.name
    counts = _split_counts(dest, spec.splits)
    if not force and _counts_complete(spec, counts, ctx.report):
        ctx.report.info(f"{spec.name}: already complete ({_format_counts(counts)}) in {dest}")
        return _Result(spec.name, True, "already complete", dest, counts, "-")

    if os.path.lexists(dest) and not os.path.islink(dest) and not dest.is_dir():
        reason = f"{dest} exists but is not a folder; move it out of the way and run again"
        ctx.report.warn(f"{spec.name}: {reason}")
        return _Result(spec.name, False, "FAILED", dest, source="-", errors=[f"local: {reason}"])
    if force and os.path.lexists(dest):
        state = "re-downloading (--force; the current files are kept until the new copy is ready)"
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


def _check_timeout(timeout: float) -> float:
    """The timeout as a float; ValueError unless it is above 0 and at most MAX_TIMEOUT."""
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be a number of seconds, got {timeout!r}") from None
    if not 0 < value <= MAX_TIMEOUT:  # also rejects NaN and infinity
        raise ValueError(
            f"timeout must be above 0 and at most {MAX_TIMEOUT:g} seconds (one day), "
            f"got {timeout!r}"
        )
    return value


def _download_datasets(
    names: Sequence[str], root: Path, force: bool, timeout: float, quiet: bool
) -> list[_Result]:
    """Process every dataset in turn; a failure does not stop the others."""
    ctx = _Context(root, _check_timeout(timeout), _Reporter(quiet))
    _remove_stale_temporaries(ctx.downloads, ctx.report)
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
    """Recount a dataset and check it against its manifest entries.

    Every listed file must be on disk with the listed sha256, and every image must be listed,
    except for the differences accepted when an official archive was downloaded (recorded in
    PROVENANCE.json, see _ArchiveSource.fetch).
    """
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
    check = _compare_with_manifest(spec, folder) if manifest else None
    if manifest and check:
        known = _known_deviations(root, spec.name)
        accepted = 0
        mismatches = []
        for rel, actual in sorted(check.mismatched.items()):
            recorded = known.get(rel)
            if recorded == actual:
                accepted += 1
            elif recorded is None:
                mismatches.append(_mismatch(rel, manifest[rel][0], actual))
            else:  # an accepted deviation that has changed since the download
                mismatches.append(
                    _mismatch(rel, recorded, actual) + "; expected = hash in PROVENANCE.json"
                )
        missing = [rel for rel in check.missing if not (rel in known and known[rel] is None)]
        accepted += len(check.missing) - len(missing)
        unlisted = [rel for rel, sha in sorted(check.unlisted.items()) if known.get(rel) != sha]
        accepted += len(check.unlisted) - len(unlisted)
        problems += mismatches[:10]
        if len(mismatches) > 10:
            problems.append(f"... and {len(mismatches) - 10} more sha256 mismatches")
        if missing:
            problems.append(
                f"{len(missing)} of the {check.total} files listed in dataset_manifest.json "
                f"are missing: {_some(missing)}"
            )
        if unlisted:
            problems.append(
                f"{len(unlisted)} images are not listed in dataset_manifest.json: {_some(unlisted)}"
            )
        hashes = (
            f"sha256 of {check.checked}/{check.total} listed files checked against "
            "dataset_manifest.json"
        )
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
    and replaces the old split folders only after success (other files in a dataset folder are
    kept). ``timeout`` (seconds, above 0 and at most one day) applies to connecting and to each
    wait for data; a download that gets less than 1 kB in that time fails as well. It does not
    limit the total time of a download. ``quiet`` hides progress output (warnings are still
    printed).

    Raises DatasetUnavailableError after processing the other datasets if every source failed
    for some dataset. The error message explains the manual download. Raises ValueError for an
    unknown dataset name or an invalid timeout.
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

    Where the manifest lists the dataset's files, each listed file must be present with its
    sha256 and each image must be listed. Differences that were accepted at download time
    (official archives whose bytes or names may differ from the mirror; recorded in
    PROVENANCE.json) do not count as failures. Prints one line per problem and returns True
    when everything checks out.
    """
    return _verify(_spec(name), _resolve_root(root), _Reporter()).ok


# ----------------------------------------------------------------------------- command line


def _positive_float(text: str) -> float:
    """argparse type of --timeout: seconds, above 0 and at most MAX_TIMEOUT."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0 < value <= MAX_TIMEOUT:  # also rejects NaN and infinity
        raise argparse.ArgumentTypeError(
            f"must be a number of seconds above 0 and at most {MAX_TIMEOUT:g} (one day)"
        )
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
        help="download again even if complete; the old files are replaced only after success",
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
        help="network timeout in seconds for connecting and for each wait for data; a download "
        "that gets less than 1 kB in SEC seconds also fails. Not a limit on the total download "
        f"time (default: 30, at most {MAX_TIMEOUT:g})",
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
        _print(f"{name} (expected {_format_counts(spec.splits)})")
        paths = " + ".join(str(downloads / a.filename) for a in drop_ins)
        _print(f"  0. drop-in archive, used first when present: {paths}")
        for number, source in enumerate(sources, 1):
            _print(f"  {number}. {source.label}")
            for slot, url, check in source.slots():
                _print(f"       {slot:<22} {url}\n       {'':<22} {check}")


def _print_summary(results: Sequence[_Result], root: Path) -> None:
    """The table printed at the end of a command-line run."""
    rows = [("dataset", "status", "images", "source")]
    rows += [(r.name, r.status, _format_counts(r.counts) or "-", r.source or "-") for r in results]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    _print(f"\nSummary (root: {root})")
    for row in rows:
        cells = (cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        _print(("  " + "  ".join(cells)).rstrip())


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    _safe_console()
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
        _write(sys.stderr, "\nInterrupted; temporary files were removed.\n")
        return 130
    _print_summary(results, root)
    failed = [r for r in results if not r.ok]
    if failed:
        _write(sys.stderr, "\n" + _unavailable_message(failed, root) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
