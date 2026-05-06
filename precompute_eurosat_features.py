#!/usr/bin/env python3
"""
Precompute frozen-backbone features for EuroSAT with **real** geographic
metadata from the georeferenced Sentinel-2 GeoTIFFs.

This script:
  1. Loads the georeferenced EuroSATallBands archive (13-channel GeoTIFFs
     with real CRS/transform metadata from Sentinel-2 tile extraction).
     The archive is obtained either via torchgeo (preferred, if installed)
     or by direct download from HuggingFace.
  2. Extracts real lat/lon (WGS84) per image from each GeoTIFF's CRS +
     affine transform via rasterio + pyproj.
  3. Clusters images into N geographic regions (nodes) with K-Means.
  4. Builds a geographic Barabási–Albert graph where connection
     probability decays with real geodesic distance.
  5. Extracts ImageNet-pretrained backbone features from Sentinel-2
     B04/B03/B02 (= red/green/blue) bands, rescaled to a photographic
     dynamic range so the RGB pretraining still transfers.
  6. Saves per-node assignments, coordinates, adjacency list, and the
     frozen features to a single .pt file.
  7. Renders a 3-panel PNG map on a real Europe basemap (cartopy) for
     visual sanity-check of the clustering and graph topology.

Outputs a single .pt file containing:
  feats_train:    [N_total, D]   float16  — backbone features
  labs_train:     [N_total]      int64    — class labels
  coords:         [N_total, 2]   float32  — (lat, lon), WGS84, real
  node_assign:    [N_total]      int64    — geographic cluster assignment
  node_centers:   [N_nodes, 2]   float32  — cluster centroids (lat, lon)
  adj_list:       dict {i: [j, ...]}      — geographic BA adjacency
  arch_map:       dict {i: str}           — per-node architecture
  class_names:    list[str]               — EuroSAT class names
  meta:           dict                    — arch/dataset/config info

Requirements:
  pip install rasterio pyproj torchvision torch scikit-learn
  # optional but recommended:
  pip install torchgeo cartopy

2026-04-21 rewrite:
  The previous version of this script generated **synthetic** European
  coordinates when it could not extract real geotags from
  torchvision.datasets.EuroSAT (which downloads only the RGB preview
  archive with geotags stripped). That fallback has been removed in
  full — research pipelines must not fabricate geographic data. The
  script now hard-requires the georeferenced EuroSATallBands archive;
  if it cannot be obtained, the script errors out rather than
  silently substituting synthetic data.

Example usage (three distinct modes):

  # (1) FIRST-EVER BUILD — extracts coords (~40min one-time) + features
  #     (~10-30min on GPU), builds graph, writes .pt and maps.
  python precompute_eurosat_features.py \\
    --arch mobilenet_v2 --num_nodes 50 --ba_m 2 --use_torchgeo \\
    --out_path ./cache/eurosat_geo_mnv2.pt

  # (2) RE-RENDER MAPS from an existing .pt (skips Steps 0-5 entirely,
  #     takes ~10s, just regenerates the three map PNGs and optional
  #     coords JSON). Errors out (exit 4) if the .pt doesn't exist.
  python precompute_eurosat_features.py \\
    --arch mobilenet_v2 --num_nodes 50 --ba_m 2 \\
    --out_path ./cache/eurosat_geo_mnv2.pt \\
    --use_cached_features

  # (3) SECOND-BACKBONE BUILD (e.g. for --random_models heterogeneous
  #     runs). Re-extracts features with a different backbone but
  #     automatically reuses the sidecar coord cache so the 40min
  #     coord pass is skipped. Keep --num_nodes / --ba_m / --seed
  #     identical to the first-backbone run so topology matches.
  python precompute_eurosat_features.py \\
    --arch efficientnet_b0 --num_nodes 50 --ba_m 2 --seed 0 \\
    --out_path ./cache/eurosat_geo_enb0.pt
"""

import argparse
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from torchvision.models import efficientnet_b0, mobilenet_v2
try:
    from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights
except Exception:
    MobileNet_V2_Weights = None
    EfficientNet_B0_Weights = None

# rasterio + pyproj are REQUIRED — no fallback.
# Previously this script had a synthetic-coord fallback path when these
# were missing; that path has been removed (2026-04-21 rewrite).
import rasterio
from rasterio.warp import transform as warp_transform


# =====================================================================
#  Constants
# =====================================================================

# EuroSAT class names in the canonical alphabetical order used by the
# original dataset's class folder layout.
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Band order inside each EuroSATallBands 13-channel GeoTIFF, matching
# torchgeo's `EuroSAT.all_band_names`. rasterio is 1-indexed, so
# B04 = band 4, B03 = band 3, B02 = band 2 — these are the Sentinel-2
# red/green/blue bands (10 m native resolution).
EUROSAT_BAND_NAMES = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B08A", "B09", "B10", "B11", "B12",
)
SENTINEL2_RGB_BANDS_1INDEXED = (4, 3, 2)  # R, G, B

# EuroSATallBands archive — 2 GB download from HuggingFace, mirror of
# the original madm.dfki.de hosting. Checksum from torchgeo upstream.
EUROSATALLBANDS_URL = (
    "https://hf.co/datasets/torchgeo/eurosat/resolve/"
    "1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/EuroSATallBands.zip"
)
EUROSATALLBANDS_FILENAME = "EuroSATallBands.zip"
EUROSATALLBANDS_MD5 = "5ac12b3b2557aa56e1826e981e8e200e"
EUROSATALLBANDS_UNPACKED_DIR = "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"

# ImageNet normalization — the backbones are pretrained on ImageNet, so
# after scaling Sentinel-2 RGB to [0, 1] we apply these stats.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Sentinel-2 L1C reflectance is stored as uint16 with quantification
# value 10000. In practice most land-cover pixels sit well below that:
# dividing by ~3000 and clipping to [0, 1] is the standard photographic
# rescaling used throughout the remote-sensing ML literature (see e.g.
# torchgeo's EuroSAT.plot(): `np.clip(image / 3000, 0, 1)`).
SENTINEL2_BRIGHTNESS_DIV = 3000.0


# =====================================================================
#  Step 0: Obtain the georeferenced archive
# =====================================================================

def _md5_file(path: str, chunk: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file, reading in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _download_with_progress(url: str, dst: str) -> None:
    """Simple progress-printing download. Streams to disk, no buffering
    of the full 2 GB archive in memory."""
    req = urllib.request.Request(url, headers={"User-Agent": "precompute-eurosat/1.0"})
    with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_report = 0
        chunk = 1 << 16
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total and downloaded - last_report > 64 * (1 << 20):
                pct = 100.0 * downloaded / total
                print(f"    {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB "
                      f"({pct:.1f}%)")
                last_report = downloaded
    print(f"    done: {downloaded / 1e9:.2f} GB")


def obtain_eurosat_allbands(
    data_root: str,
    *,
    use_torchgeo: bool = False,
    verify_md5: bool = True,
    force_extract: bool = False,
) -> str:
    """Ensure EuroSATallBands is present on disk and return its root dir.

    2026-04-21 rewrite: this replaces the old torchvision-based loader,
    which silently downloaded the geotag-stripped RGB JPEG archive and
    left the script unable to extract real coordinates. We now insist
    on the georeferenced TIFF archive.

    2026-04-22 revision: the previous "already unpacked" check returned
    early on ANY tif files found (`_count_tifs > 0`). That incorrectly
    accepted partial extractions — if a previous run was interrupted
    after unpacking only a few class folders, the loader happily used
    whatever was there and later crashed with "Missing class folders".
    The new check requires all 10 classes present with realistic file
    counts, re-extracts from the cached zip when possible, and surfaces
    per-class counts so the user can see exactly what's wrong.

    Args:
        data_root: parent directory where the dataset should live.
        use_torchgeo: if True, let torchgeo handle download & verification.
            Requires `pip install torchgeo`.
        verify_md5: if True and we downloaded directly, verify the MD5
            against the known upstream value.
        force_extract: if True, re-extract from the cached zip even if
            the class folders look complete. Useful when a previous
            extraction was interrupted and partial files are present.

    Returns:
        Absolute path to the directory containing the per-class TIFF
        subfolders (AnnualCrop/, Forest/, ...).
    """
    os.makedirs(data_root, exist_ok=True)

    # Expected final layout: <data_root>/<EUROSATALLBANDS_UNPACKED_DIR>/<class>/*.tif
    final_dir = os.path.join(data_root, EUROSATALLBANDS_UNPACKED_DIR)

    # 2026-04-22: if the expected root doesn't exist, search for a
    # working one in case a prior extraction landed elsewhere.
    if not os.path.isdir(final_dir):
        hits = _find_class_folder_root(data_root)
        if hits:
            final_dir = hits

    # Verify completeness: all 10 classes present with plausible counts.
    if not force_extract and os.path.isdir(final_dir):
        complete, per_class, total = _verify_unpacked(final_dir)
        if complete:
            print(f"[DATA] EuroSATallBands already unpacked at {final_dir} "
                  f"({total} .tif files across 10 classes).")
            return final_dir
        else:
            print(f"[DATA] PARTIAL extraction detected at {final_dir}:")
            _print_per_class_counts(per_class)
            print(f"[DATA] Will re-extract from zip (pass "
                  f"--skip_repair to bypass this behaviour).")

    # --- torchgeo path ------------------------------------------------
    if use_torchgeo:
        try:
            from torchgeo.datasets import EuroSAT as _TGEuroSAT  # noqa
        except ImportError:
            print("[DATA] --use_torchgeo requested but torchgeo is not "
                  "installed. Install with `pip install torchgeo` or "
                  "re-run without --use_torchgeo for direct download.",
                  file=sys.stderr)
            sys.exit(2)
        print("[DATA] Using torchgeo to fetch / verify EuroSATallBands...")
        # 2026-04-22: if the unpacked tree is already partial, torchgeo
        # will happily see the base_dir exists and skip re-extraction —
        # same bug as my own check before this revision. Nuke the
        # partial tree first so torchgeo actually does the work.
        if os.path.isdir(final_dir):
            complete, _, _ = _verify_unpacked(final_dir)
            if not complete or force_extract:
                print(f"[DATA] Removing partial tree at {final_dir} so "
                      f"torchgeo re-extracts cleanly.")
                shutil.rmtree(final_dir)
        # Instantiate once per split to trigger download + extraction.
        for split in ("train", "val", "test"):
            _TGEuroSAT(root=data_root, split=split, download=True,
                       checksum=verify_md5)
        # Re-locate final_dir after torchgeo's extraction.
        if not os.path.isdir(final_dir):
            hits = _find_class_folder_root(data_root)
            if hits:
                final_dir = hits
        if not os.path.isdir(final_dir):
            raise RuntimeError(
                f"torchgeo extracted the archive but the expected "
                f"class-folder root could not be located under {data_root}.")
        _assert_unpacked(final_dir)
        return final_dir

    # --- direct download path ----------------------------------------
    zip_path = os.path.join(data_root, EUROSATALLBANDS_FILENAME)

    need_download = not os.path.isfile(zip_path)
    if not need_download and verify_md5:
        got = _md5_file(zip_path)
        if got != EUROSATALLBANDS_MD5:
            print(f"[DATA] Cached zip MD5 mismatch ({got} vs expected "
                  f"{EUROSATALLBANDS_MD5}); re-downloading.")
            need_download = True

    if need_download:
        print(f"[DATA] Downloading EuroSATallBands (~2 GB) from "
              f"{EUROSATALLBANDS_URL} ...")
        _download_with_progress(EUROSATALLBANDS_URL, zip_path)
        if verify_md5:
            got = _md5_file(zip_path)
            if got != EUROSATALLBANDS_MD5:
                raise RuntimeError(
                    f"EuroSATallBands.zip MD5 mismatch: "
                    f"expected {EUROSATALLBANDS_MD5}, got {got}. "
                    f"The upstream archive may have moved; re-check the URL.")
            print(f"[DATA] MD5 verified: {got}")
    else:
        print(f"[DATA] Using existing archive at {zip_path}")

    # 2026-04-22: before extracting, if a partial tree exists wipe the
    # class folders that ARE present so the zip's files overwrite
    # cleanly and we don't get half-and-half state. We only nuke the
    # per-class subdirs, not the whole data_root (which would also
    # delete the multi-GB zip we just downloaded).
    if os.path.isdir(final_dir) and not force_extract:
        complete, _, _ = _verify_unpacked(final_dir)
        if not complete:
            print(f"[DATA] Removing partial class subfolders at "
                  f"{final_dir} before re-extracting...")
            for cls in EUROSAT_CLASSES:
                p = os.path.join(final_dir, cls)
                if os.path.isdir(p):
                    shutil.rmtree(p)

    print(f"[DATA] Extracting archive into {data_root} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_root)

    if not os.path.isdir(final_dir):
        hits = _find_class_folder_root(data_root)
        if hits:
            final_dir = hits
    if not os.path.isdir(final_dir):
        raise RuntimeError(
            f"After extraction, the expected class-folder root was not "
            f"found under {data_root}. Archive layout may have changed.")

    _assert_unpacked(final_dir)
    complete, per_class, total = _verify_unpacked(final_dir)
    print(f"[DATA] Extracted {total} .tif files across 10 classes.")
    return final_dir


# ---------------------------------------------------------------------
# 2026-04-22: per-class verification helpers. The old `_count_tifs`
# returned a single integer, which wasn't enough to detect partial
# extractions where only some classes had unpacked. The new helpers
# surface per-class counts so we can fail fast with a useful message
# and repair automatically.
# ---------------------------------------------------------------------

# Rough expected per-class counts from the EuroSAT paper. We don't
# require an exact match (any unused file has no reason to be missing),
# but per-class count ≥ this floor catches "only got a few stragglers"
# partial extractions.
_MIN_FILES_PER_CLASS = 100


def _per_class_counts(root: str) -> Dict[str, int]:
    """Return {class_name: num_tif_files} for each EuroSAT class, or 0
    if the class folder is missing entirely. Always emits all 10 keys."""
    out = {}
    for cls in EUROSAT_CLASSES:
        cls_dir = os.path.join(root, cls)
        if os.path.isdir(cls_dir):
            out[cls] = len(glob.glob(os.path.join(cls_dir, "*.tif")))
        else:
            out[cls] = 0
    return out


def _verify_unpacked(root: str) -> Tuple[bool, Dict[str, int], int]:
    """Return (complete, per_class, total).

    `complete` is True iff all 10 class folders exist with at least
    `_MIN_FILES_PER_CLASS` .tif files each. Used to distinguish a clean
    extraction from a partial/corrupted one.
    """
    per_class = _per_class_counts(root)
    complete = all(v >= _MIN_FILES_PER_CLASS for v in per_class.values())
    return complete, per_class, sum(per_class.values())


def _print_per_class_counts(per_class: Dict[str, int]) -> None:
    """Pretty-print per-class file counts, marking deficient classes."""
    for cls in EUROSAT_CLASSES:
        n = per_class.get(cls, 0)
        tag = "OK " if n >= _MIN_FILES_PER_CLASS else "BAD"
        print(f"    [{tag}] {cls:25s}: {n:5d}")


def _assert_unpacked(root: str) -> None:
    """Raise with a clear message if the tree at `root` isn't a complete
    EuroSATallBands extraction."""
    complete, per_class, total = _verify_unpacked(root)
    if complete:
        return
    print(f"[DATA] Extraction verification FAILED for {root}:",
          file=sys.stderr)
    _print_per_class_counts(per_class)
    missing = [c for c, n in per_class.items() if n == 0]
    short = [c for c, n in per_class.items()
             if 0 < n < _MIN_FILES_PER_CLASS]
    hints = []
    if missing:
        hints.append(f"missing classes: {missing}")
    if short:
        hints.append(
            f"under-filled classes (<{_MIN_FILES_PER_CLASS} files): {short}")
    raise RuntimeError(
        f"EuroSATallBands extraction under {root} is not complete "
        f"({total} files total). " + "; ".join(hints) +
        ". Re-run with --force_extract to nuke and re-extract, or "
        "delete the directory manually.")


def _find_class_folder_root(root: str) -> Optional[str]:
    """Walk downward to find a directory containing all 10 EuroSAT class
    subfolders. Handles variations in how different archives unpack."""
    for dirpath, dirnames, _ in os.walk(root):
        if all(cls in dirnames for cls in EUROSAT_CLASSES):
            return dirpath
    return None


# =====================================================================
#  Step 1: Dataset with real georeferenced coordinates
# =====================================================================

# 2026-04-22: progress helper for the parallel coord extraction. Kept
# at module scope so the extraction routine can call it without tying
# progress rendering to dataset-instance state.

def _print_coord_progress(done: int, total: int, t0: float) -> None:
    """Pretty progress line with rate + ETA, flushed immediately so it
    actually appears under stdout buffering (tee, log redirects, SLURM).
    """
    elapsed = max(1e-6, time.time() - t0)
    rate = done / elapsed
    remaining = (total - done) / max(rate, 1e-6)
    pct = 100.0 * done / max(total, 1)
    print(f"  [COORDS] {done:>6}/{total}  ({pct:5.1f}%)  "
          f"{rate:5.0f} tif/s  elapsed {elapsed:6.1f}s  "
          f"ETA {remaining:6.1f}s",
          flush=True)


# ---------------------------------------------------------------------
# 2026-04-22: PROJ environment validator.
#
# Added after a real failure: a user ran the script for 40 minutes
# and had all 27 000 files fail with the same CRSError —
#     PROJ: ... proj.db contains DATABASE.LAYOUT.VERSION.MINOR = 4
#     whereas a number >= 6 is expected. It comes from another
#     PROJ installation.
# — a classic conda/pip PROJ conflict (old conda proj.db picked up by
# newer pip-installed rasterio/pyproj).
#
# 2026-04-22 (second revision, post-attempted-fix): even after the user
# tried the suggested remediation (unset PROJ_*, pip reinstall),
# conda's activation scripts kept re-setting PROJ_DATA behind their
# back, so the bad proj.db was still being picked up. So this
# validator now *probes* several transformation modes and picks
# whichever one works, in priority order:
#
#   Mode A: rasterio.warp with EPSG codes on both sides
#     — the standard path. Needs a working proj.db.
#   Mode B: rasterio.warp with a PROJ-string destination
#     — destination is parsed as a proj4 string, no DB lookup.
#       Still needs the source CRS to be resolvable, but WKT from
#       the TIFF is often enough.
#   Mode C: pyproj.Transformer with WKT source + PROJ-string dest
#     — both sides fully self-describing; bypasses the EPSG DB
#       entirely. Works even with a broken proj.db as long as the
#       PROJ *core* library can still parse WKT (which it usually can).
#
# The chosen mode is cached in _TRANSFORM_MODE so per-file calls use
# the same path. If ALL THREE fail, we print setup guidance and exit.
# ---------------------------------------------------------------------

# Populated by _validate_proj_env() at startup. One of "epsg",
# "proj_string_dst", "pyproj_wkt". Used by _read_latlon_via_mode().
_TRANSFORM_MODE: Optional[str] = None

# Canonical WGS84 PROJ string — parsed directly by PROJ without any
# database lookup. Used as the destination in modes B and C.
_WGS84_PROJ_STRING = "+proj=longlat +datum=WGS84 +no_defs +type=crs"


def _probe_epsg_mode() -> Tuple[float, float]:
    """Mode A: EPSG code → EPSG code via rasterio.warp."""
    lons, lats = warp_transform(
        rasterio.crs.CRS.from_epsg(32634),
        "EPSG:4326",
        [739454.0], [4207432.0])
    return float(lats[0]), float(lons[0])


def _probe_proj_string_dst_mode() -> Tuple[float, float]:
    """Mode B: rasterio.warp with a PROJ-string destination.
    Still uses rasterio, but destination doesn't hit the EPSG DB.
    Source CRS is constructed from WKT rather than EPSG code."""
    src_wkt = rasterio.crs.CRS.from_epsg(32634).to_wkt()
    lons, lats = warp_transform(
        rasterio.crs.CRS.from_wkt(src_wkt),
        rasterio.crs.CRS.from_string(_WGS84_PROJ_STRING),
        [739454.0], [4207432.0])
    return float(lats[0]), float(lons[0])


def _probe_pyproj_wkt_mode() -> Tuple[float, float]:
    """Mode C: pyproj.Transformer with WKT source + PROJ-string dest.
    Fully bypasses rasterio.warp AND the EPSG database."""
    import pyproj
    # We still need to get the source WKT somehow. For the probe, we
    # construct it from the EPSG code (which *does* need the DB). But
    # this is fine because in practice `_read_latlon` reads the WKT
    # directly from the TIFF file, not from an EPSG lookup. The probe
    # is only checking whether the *transform* path works end-to-end.
    try:
        src_wkt = rasterio.crs.CRS.from_epsg(32634).to_wkt()
    except Exception:
        # If EPSG→WKT also fails, use a hard-coded WKT for UTM 34N
        # so the probe can still complete.
        src_wkt = (
            'PROJCS["WGS 84 / UTM zone 34N",'
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],'
            'PARAMETER["central_meridian",21],'
            'PARAMETER["scale_factor",0.9996],'
            'PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1]]'
        )
    trans = pyproj.Transformer.from_crs(
        src_wkt, _WGS84_PROJ_STRING, always_xy=True)
    lon, lat = trans.transform(739454.0, 4207432.0)
    return float(lat), float(lon)


def _validate_proj_env() -> None:
    """Probe the three transformation modes and pick the first that works.

    The chosen mode is stashed in the module-level _TRANSFORM_MODE so
    every `_read_latlon` call uses the same path (no per-file probing).
    """
    global _TRANSFORM_MODE

    probes = [
        ("epsg",            _probe_epsg_mode,
         "rasterio.warp with EPSG codes (standard path)"),
        ("proj_string_dst", _probe_proj_string_dst_mode,
         "rasterio.warp with PROJ-string destination (skips DB on dst)"),
        ("pyproj_wkt",      _probe_pyproj_wkt_mode,
         "pyproj.Transformer with WKT + PROJ-string (skips EPSG DB entirely)"),
    ]

    attempts: List[Tuple[str, str]] = []
    for name, fn, description in probes:
        try:
            lat, lon = fn()
            if not (35.0 < lat < 41.0 and 20.0 < lon < 27.0):
                raise RuntimeError(
                    f"self-test returned wrong coords "
                    f"(lat={lat}, lon={lon}); expected near Athens")
            _TRANSFORM_MODE = name
            label = "STANDARD" if name == "epsg" else "FALLBACK"
            print(f"[COORDS] PROJ self-test: mode='{name}' works "
                  f"[{label}] — {description}",
                  flush=True)
            if name != "epsg":
                print(f"[COORDS] NOTE: using a fallback path because the "
                      f"standard EPSG-based transform is broken on this "
                      f"system. Coordinates will still be correct; "
                      f"consider fixing your PROJ install when convenient.",
                      flush=True)
            return
        except Exception as e:
            attempts.append((name, f"{e.__class__.__name__}: {e}"))

    # All three failed — print the full setup-guidance banner.
    print("", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("  PROJ ENVIRONMENT CHECK FAILED", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    for name, err in attempts:
        print(f"  mode={name!r} failed:", file=sys.stderr)
        print(f"    {err}", file=sys.stderr)
    print("", file=sys.stderr)
    is_proj_db_mismatch = any(
        "proj.db" in err and "LAYOUT.VERSION" in err
        for _, err in attempts)

    if is_proj_db_mismatch:
        print("  This is a conda/pip PROJ version mismatch. Fixing it",
              file=sys.stderr)
        print("  requires working around conda's activation scripts,",
              file=sys.stderr)
        print("  which usually re-export PROJ_DATA even after 'unset'.",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("  FIX OPTIONS — try these in order:", file=sys.stderr)
        print("", file=sys.stderr)
        print("  (A) Upgrade conda's proj to match pyproj/rasterio "
              "(most reliable):", file=sys.stderr)
        print("        conda install -n Move -c conda-forge --force-reinstall \\",
              file=sys.stderr)
        print("              'proj>=9' 'pyproj>=3.6' 'rasterio>=1.3'",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("  (B) Replace the stale conda proj.db with pyproj's "
              "bundled one:", file=sys.stderr)
        print("        PYPROJ_DB=$(python -c "
              "\"import pyproj.datadir as d; print(d.get_data_dir())\")",
              file=sys.stderr)
        print("        # back up and swap:", file=sys.stderr)
        print("        mv  ./miniconda3/envs/Move/"
              "share/proj/proj.db \\", file=sys.stderr)
        print("            ./miniconda3/envs/Move/"
              "share/proj/proj.db.bak", file=sys.stderr)
        print("        cp  \"$PYPROJ_DB/proj.db\" \\", file=sys.stderr)
        print("            ./miniconda3/envs/Move/"
              "share/proj/proj.db", file=sys.stderr)
        print("", file=sys.stderr)
        print("  (C) Point PROJ at pyproj's bundled data every shell:",
              file=sys.stderr)
        print("        export PROJ_DATA=\"$(python -c "
              "'import pyproj.datadir; print(pyproj.datadir.get_data_dir())')\"",
              file=sys.stderr)
        print("        python precompute_eurosat_features.py ...",
              file=sys.stderr)
        print("      NOTE: conda's activate.d scripts may re-set",
              file=sys.stderr)
        print("      PROJ_DATA on env activation. Check with:",
              file=sys.stderr)
        print("        grep -r PROJ_DATA "
              "./miniconda3/envs/Move/etc/",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("  DIAGNOSTIC: show what PROJ currently sees:",
              file=sys.stderr)
        print("        python -c 'from pyproj import show_versions; "
              "show_versions()'", file=sys.stderr)
    else:
        print("  Rasterio / pyproj appear to be misconfigured. Confirm",
              file=sys.stderr)
        print("  you can import both:", file=sys.stderr)
        print('        python -c "import rasterio, pyproj; '
              'print(rasterio.__version__, pyproj.__version__)"',
              file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    sys.exit(3)


# ---------------------------------------------------------------------
# 2026-04-22: coord cache.
#
# The coord extraction pass takes ~40 minutes on 27k files. Once it
# succeeds, there is no reason to redo it on subsequent runs — the
# results are deterministic and only depend on the underlying GeoTIFFs.
# We cache the (path → (lat, lon)) mapping in a sidecar JSON file next
# to the tif tree. On rerun, if the cache contains all current sample
# paths, we short-circuit the extraction pass.
# ---------------------------------------------------------------------

_COORD_CACHE_FILENAME = "_eurosat_coord_cache.json"
_COORD_CACHE_VERSION = 1  # bump if the cache schema changes


def _coord_cache_path(tif_root: str) -> str:
    return os.path.join(tif_root, _COORD_CACHE_FILENAME)


def _load_coord_cache(tif_root: str) -> Optional[Dict[str, List[float]]]:
    """Return {basename: [lat, lon]} or None if cache is missing/unusable."""
    path = _coord_cache_path(tif_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("version") != _COORD_CACHE_VERSION:
            print(f"[COORDS] Ignoring coord cache with version "
                  f"{data.get('version')} != {_COORD_CACHE_VERSION}.",
                  flush=True)
            return None
        coords = data.get("coords", {})
        if not isinstance(coords, dict):
            return None
        return coords
    except Exception as e:
        print(f"[COORDS] Could not read coord cache {path}: {e}. "
              f"Will re-extract.", flush=True)
        return None


def _save_coord_cache(tif_root: str,
                      coords_by_basename: Dict[str, List[float]]) -> None:
    """Persist the coord map so reruns skip the extraction pass."""
    path = _coord_cache_path(tif_root)
    tmp = path + ".tmp"
    blob = {
        "version": _COORD_CACHE_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_entries": len(coords_by_basename),
        "coords": coords_by_basename,
    }
    try:
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, path)
        print(f"[COORDS] Cached {len(coords_by_basename)} coords to {path} "
              f"(reruns will skip the extraction pass).",
              flush=True)
    except Exception as e:
        print(f"[COORDS] WARNING: Could not write coord cache {path}: {e}",
              file=sys.stderr)
        # Best-effort cleanup
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


class EuroSATWithCoords(Dataset):
    """EuroSAT dataset that returns real (lat, lon) alongside each image.

    Reads the georeferenced 13-channel GeoTIFFs from EuroSATallBands.
    For each file, the image centre pixel is converted from the file's
    native CRS (Sentinel-2 uses UTM zones for L1C tiles) to WGS84
    lat/lon via rasterio's `warp_transform`.

    2026-04-21 rewrite: the previous implementation had a synthetic-
    coordinate fallback for when geotags were unavailable. That has
    been removed — any file that fails to yield valid coordinates
    is a hard error here, because research pipelines must not
    fabricate geographic data.
    """

    def __init__(self, tif_root: str, transform=None,
                 require_all_valid: bool = True,
                 use_coord_cache: bool = True):
        # 2026-04-22: use_coord_cache lets callers disable the sidecar
        # cache (e.g. if they suspect corruption and want a clean pass).
        self.tif_root = tif_root
        self.transform = transform
        self.use_coord_cache = use_coord_cache

        # Build (path, class_idx) list
        self.samples: List[Tuple[str, int, float, float]] = []
        missing_classes = []
        for class_idx, cls in enumerate(EUROSAT_CLASSES):
            cls_dir = os.path.join(tif_root, cls)
            if not os.path.isdir(cls_dir):
                missing_classes.append(cls)
                continue
            tif_paths = sorted(glob.glob(os.path.join(cls_dir, "*.tif")))
            for p in tif_paths:
                # coords filled in during extraction pass below
                self.samples.append((p, class_idx, float("nan"), float("nan")))
        if missing_classes:
            raise RuntimeError(
                f"Missing class folders in {tif_root}: {missing_classes}")
        if not self.samples:
            raise RuntimeError(f"No .tif files found under {tif_root}.")

        # Extract coordinates up front (single rasterio pass).
        self._extract_real_coords(require_all_valid=require_all_valid)

    def _extract_real_coords(self, require_all_valid: bool) -> None:
        """Read center-pixel lat/lon from each GeoTIFF.

        This is the only way images get coordinates now — there is no
        synthetic fallback. If any file is unreadable or has no CRS,
        we error out (unless `require_all_valid=False`, in which case
        we print a warning and drop the offending samples; still no
        fabrication).

        2026-04-22: parallelized across CPU cores. Reading each TIFF's
        header + doing a single-point CRS reprojection takes ~5–20ms
        per file sequentially, which for 27k files is a 2–9 minute wall
        clock with no feedback. The old version only printed progress
        every 5000 files, so users could easily think it was hung.
        Now we:
          (a) use a ThreadPoolExecutor to parallelize the work — each
              thread opens its own rasterio handle (safe) and GDAL
              releases the GIL during file I/O, so threads give near-
              linear speedup without the pickling / fork-after-GDAL
              complications that ProcessPool runs into on some setups;
          (b) print a progress line every 2 seconds with rate + ETA so
              the user can see it's alive and estimate remaining time;
          (c) flush stdout on every print so the messages actually
              reach the terminal when stdout is line-buffered (e.g.
              under `tee`, `> logfile`, or on some SLURM setups).

        2026-04-22 (post-incident): added three defensive layers after
        a user hit a broken PROJ install, ran the script for 40 minutes,
        and had all 27 000 files fail with the same CRSError —
            (1) sidecar coord cache: on rerun we look up basename →
                (lat, lon) from _eurosat_coord_cache.json and skip the
                rasterio pass entirely when the cache covers every
                current sample. That 40 minutes of work is never wasted
                again once it has succeeded once.
            (2) PROJ env self-test: before doing ANY work we do a
                single known-good UTM→WGS84 reprojection. If PROJ is
                broken (e.g. conda/pip proj.db mismatch) we abort
                within a second with a clear diagnostic, instead of
                plowing through 27k files.
            (3) systematic-failure early-abort: if the first N files
                all fail with the same exception class, something is
                wrong with the environment / data / code, and there's
                no point continuing. We stop after ~50 failures in a
                row and surface the problem.
        """
        import concurrent.futures

        n_total = len(self.samples)

        # --- Layer 1: try the sidecar cache first ------------------------
        if self.use_coord_cache:
            cache = _load_coord_cache(self.tif_root)
            if cache is not None:
                # Check coverage: is every current sample present in the
                # cache? We key on basename because the absolute tif_root
                # may change between runs (symlinks, moved dataset dir,
                # different mount point), but the relative layout is
                # stable and each EuroSAT .tif filename is globally
                # unique across the archive.
                missing = 0
                filled = []
                for (path, cls_idx, _, _) in self.samples:
                    key = os.path.basename(path)
                    ent = cache.get(key)
                    if ent is None or len(ent) != 2:
                        missing += 1
                        filled.append((path, cls_idx, float("nan"),
                                       float("nan")))
                    else:
                        filled.append((path, cls_idx,
                                       float(ent[0]), float(ent[1])))
                if missing == 0:
                    self.samples = filled
                    print(f"[COORDS] Using cached coordinates for all "
                          f"{n_total} files from "
                          f"{_coord_cache_path(self.tif_root)} — "
                          f"skipping extraction pass.",
                          flush=True)
                    lats = np.array([s[2] for s in self.samples],
                                    dtype=np.float64)
                    lons = np.array([s[3] for s in self.samples],
                                    dtype=np.float64)
                    print(f"[COORDS] Real lat/lon extent: "
                          f"lat=[{lats.min():.3f}, {lats.max():.3f}]  "
                          f"lon=[{lons.min():.3f}, {lons.max():.3f}]",
                          flush=True)
                    return
                else:
                    print(f"[COORDS] Cache found but {missing} / {n_total} "
                          f"entries missing; will re-extract in full.",
                          flush=True)

        # --- Layer 2: PROJ environment self-test BEFORE the big pass ----
        # If PROJ is broken we want to know in one second, not 40 minutes.
        _validate_proj_env()

        # Default worker count: leave two cores free for the OS / I/O,
        # cap at 16 because EuroSAT TIFFs are small and past that the
        # rasterio / GDAL lock overhead dominates any extra parallelism.
        default_workers = max(1, min(16, (os.cpu_count() or 4) - 2))
        n_workers = int(os.environ.get(
            "EUROSAT_COORD_WORKERS", default_workers))
        print(f"[COORDS] Extracting real lat/lon from {n_total} GeoTIFFs "
              f"(one-time pass, using {n_workers} parallel workers)...",
              flush=True)

        t0 = time.time()
        last_report = t0
        paths_only = [s[0] for s in self.samples]
        results: List[Optional[Tuple[float, float, Optional[str]]]] = [
            None] * n_total
        done = 0
        report_every_sec = 2.0

        # --- Layer 3: systematic-failure early abort --------------------
        # If the first ~50 consecutive files all fail with the same
        # error class (e.g. every file raises CRSError), something is
        # fundamentally wrong with the environment — keep going and
        # you'll just reproduce the same failure 27k times. We raise a
        # concise error from the loop as soon as we hit the threshold.
        EARLY_ABORT_THRESHOLD = 50
        consecutive_failures = 0
        first_error_sample: Optional[Tuple[str, str]] = None  # (path, err)
        abort_error: Optional[str] = None

        def _on_result(idx: int, res) -> bool:
            """Update progress / early-abort state. Returns True to stop."""
            nonlocal done, last_report, consecutive_failures
            nonlocal first_error_sample, abort_error
            results[idx] = res
            done += 1
            _lat, _lon, err = res
            if err is not None:
                if first_error_sample is None:
                    first_error_sample = (paths_only[idx], err)
                consecutive_failures += 1
                if consecutive_failures >= EARLY_ABORT_THRESHOLD:
                    abort_error = err
                    return True
            else:
                consecutive_failures = 0
            now = time.time()
            if now - last_report >= report_every_sec or done == n_total:
                _print_coord_progress(done, n_total, t0)
                last_report = now
            return False

        if n_workers <= 1:
            # Sequential path — simpler to debug, no thread overhead.
            for idx, path in enumerate(paths_only):
                res = self._read_latlon(path)
                if _on_result(idx, res):
                    break
        else:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_workers) as ex:
                # ex.map preserves input order; we use it so results[idx]
                # stays aligned with class labels.
                stop = False
                it = ex.map(self._read_latlon, paths_only)
                for idx, res in enumerate(it):
                    if _on_result(idx, res):
                        stop = True
                        break
                if stop:
                    # Best-effort cancel of remaining work. The pool is
                    # already iterating, so we just drain via shutdown.
                    ex.shutdown(wait=False, cancel_futures=True)

        if abort_error is not None:
            # Systematic failure — bail out with a clear diagnostic
            # instead of finishing and printing "27000 failures".
            first_path, first_err = (first_error_sample
                                     if first_error_sample
                                     else (paths_only[0], abort_error))
            raise RuntimeError(
                f"Aborted after {EARLY_ABORT_THRESHOLD} consecutive coord "
                f"failures — this looks systematic, not per-file.\n"
                f"  First failing file: {first_path}\n"
                f"  Error: {first_err}\n"
                f"This is almost always a PROJ / rasterio install issue. "
                f"Scroll up for remediation steps printed by the PROJ "
                f"self-test (if it fired). If the self-test passed but "
                f"extraction still fails systematically, try re-installing "
                f"rasterio+pyproj from conda-forge.")

        # Assemble results (for any missed entries from the early break,
        # results[idx] stays None; we treat those as failures).
        updated = []
        bad = []
        coords_cache_out: Dict[str, List[float]] = {}
        for (path, cls_idx, _, _), res in zip(self.samples, results):
            if res is None:
                bad.append((path, "not processed (early abort)"))
                continue
            lat, lon, err = res
            if err is not None:
                bad.append((path, err))
                continue
            updated.append((path, cls_idx, lat, lon))
            coords_cache_out[os.path.basename(path)] = [lat, lon]

        dt = time.time() - t0
        print(f"[COORDS] Parsed {len(updated)} / {n_total} files "
              f"in {dt:.1f}s ({len(bad)} failures).",
              flush=True)

        if bad:
            # Show first few failures so the user can diagnose.
            for path, err in bad[:5]:
                print(f"  BAD: {path}  ({err})", file=sys.stderr)
            if require_all_valid:
                raise RuntimeError(
                    f"Failed to read real coordinates from {len(bad)} "
                    f"files. Aborting — research pipelines must not "
                    f"substitute synthetic coordinates. Investigate the "
                    f"files above, or re-run with "
                    f"--allow_missing_coords to drop them.")
            else:
                print(f"[COORDS] --allow_missing_coords set — dropping "
                      f"{len(bad)} unreadable files.")

        self.samples = updated

        # 2026-04-22: persist the successful coord map so reruns skip
        # the extraction pass entirely. Only write the cache when we
        # actually produced a meaningful result — don't pollute the
        # cache file after a failed run.
        if self.use_coord_cache and coords_cache_out:
            _save_coord_cache(self.tif_root, coords_cache_out)

        # Summary of real coordinate extent.
        lats = np.array([s[2] for s in self.samples], dtype=np.float64)
        lons = np.array([s[3] for s in self.samples], dtype=np.float64)
        print(f"[COORDS] Real lat/lon extent: "
              f"lat=[{lats.min():.3f}, {lats.max():.3f}]  "
              f"lon=[{lons.min():.3f}, {lons.max():.3f}]",
              flush=True)

    @staticmethod
    def _read_latlon(tif_path: str) -> Tuple[float, float, Optional[str]]:
        """Return (lat, lon, error). On success error is None.

        Sentinel-2 L1C tiles are stored in UTM projections. rasterio
        reads the native CRS + affine transform from the TIFF; we take
        the centre pixel's projected coordinate and reproject to WGS84.

        2026-04-22 (second revision): dispatches on the module-level
        `_TRANSFORM_MODE` chosen by `_validate_proj_env()`. Mode 'epsg'
        is the standard path. Modes 'proj_string_dst' and 'pyproj_wkt'
        are fallbacks that bypass the EPSG database (needed when the
        user's proj.db is the wrong version) — they read the source
        CRS as WKT straight out of the TIFF, so no DB lookup is
        required for the source either.
        """
        try:
            with rasterio.open(tif_path) as ds:
                if ds.crs is None:
                    return (float("nan"), float("nan"),
                            "no CRS in file")
                if ds.transform == rasterio.Affine.identity():
                    return (float("nan"), float("nan"),
                            "identity transform (no real georef)")
                cx, cy = ds.xy(ds.height // 2, ds.width // 2)

                mode = _TRANSFORM_MODE or "epsg"
                if mode == "pyproj_wkt":
                    # Fully bypass rasterio.warp AND the EPSG DB.
                    import pyproj
                    src_wkt = ds.crs.to_wkt()
                    trans = pyproj.Transformer.from_crs(
                        src_wkt, _WGS84_PROJ_STRING, always_xy=True)
                    lon, lat = trans.transform(cx, cy)
                elif mode == "proj_string_dst":
                    # rasterio.warp with PROJ-string destination.
                    # Source comes directly from the file's CRS, which
                    # rasterio already read as WKT — no DB lookup needed.
                    lons, lats = warp_transform(
                        ds.crs,
                        rasterio.crs.CRS.from_string(_WGS84_PROJ_STRING),
                        [cx], [cy])
                    lat, lon = float(lats[0]), float(lons[0])
                else:
                    # mode == "epsg" — the standard path.
                    lons, lats = warp_transform(
                        ds.crs, "EPSG:4326", [cx], [cy])
                    lat, lon = float(lats[0]), float(lons[0])

                lat, lon = float(lat), float(lon)
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    return (float("nan"), float("nan"),
                            f"out-of-earth coordinates ({lat}, {lon})")
                return (lat, lon, None)
        except Exception as e:
            return (float("nan"), float("nan"),
                    f"{e.__class__.__name__}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Kept for compatibility, but feature extraction uses the
        # IndexedDataset wrapper below, which pulls the image via
        # rasterio rather than torchvision's loader.
        path, cls_idx, lat, lon = self.samples[idx]
        with rasterio.open(path) as ds:
            rgb = _read_sentinel2_rgb(ds)
        if self.transform is not None:
            rgb = self.transform(rgb)
        return rgb, cls_idx, lat, lon


def _read_sentinel2_rgb(ds) -> torch.Tensor:
    """Read a 3-channel RGB tensor from an open rasterio GeoTIFF.

    Pulls Sentinel-2 bands B04 (red), B03 (green), B02 (blue), rescales
    reflectance to [0, 1] via the standard /3000 clip, and returns a
    float32 CHW tensor ready for torchvision transforms.
    """
    # rasterio returns (bands, H, W)
    arr = ds.read(list(SENTINEL2_RGB_BANDS_1INDEXED)).astype(np.float32)
    arr = np.clip(arr / SENTINEL2_BRIGHTNESS_DIV, 0.0, 1.0)
    return torch.from_numpy(arr)  # CHW float32 in [0, 1]


# =====================================================================
#  Step 2: Geographic clustering (real coords only)
# =====================================================================

def cluster_by_geography(
    coords: np.ndarray,
    num_nodes: int,
    seed: int = 0,
    min_per_node: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster real (lat, lon) pairs into geographic regions via K-Means.

    2026-04-21 rewrite: simplified now that coords are guaranteed real.
    No NaN imputation, no synthetic-coord tolerance — if any coord is
    non-finite at this point it's a bug upstream, so we raise.
    """
    from sklearn.cluster import KMeans

    if not np.isfinite(coords).all():
        n_bad = int(np.sum(~np.isfinite(coords).all(axis=1)))
        raise ValueError(
            f"cluster_by_geography got {n_bad} non-finite coordinate "
            f"rows. This should not happen: EuroSATWithCoords errors "
            f"out on unreadable files. Check the dataset loader.")

    if coords.shape[0] < num_nodes * min_per_node:
        print(f"[WARN] Only {coords.shape[0]} coordinates for "
              f"{num_nodes} nodes (min {min_per_node} each). "
              f"Cluster sizes will be uneven.")

    # Equal-area approximation: scale lon by cos(mean_lat) so Euclidean
    # distances in the (scaled-lon, lat) plane approximate great-circle
    # distances well enough for a flat K-Means to give sensible regions.
    mean_lat = coords[:, 0].mean()
    scale = np.array([1.0, np.cos(np.radians(mean_lat))])
    coords_scaled = coords * scale

    km = KMeans(n_clusters=num_nodes, random_state=seed, n_init=10)
    assignments = km.fit_predict(coords_scaled)
    centers = km.cluster_centers_ / scale

    sizes = np.bincount(assignments, minlength=num_nodes)
    print(f"[CLUSTER] {num_nodes} nodes: "
          f"min={sizes.min()} mean={sizes.mean():.0f} "
          f"max={sizes.max()} total={len(assignments)}")
    return assignments, centers


def print_cluster_stats(
    assignments: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    num_nodes: int,
    top_k: int = 5,
):
    """Print class distribution for top-k largest clusters."""
    sizes = np.bincount(assignments, minlength=num_nodes)
    top_nodes = np.argsort(-sizes)[:top_k]
    print(f"\n[CLUSTER] Class distributions for {top_k} largest nodes:")
    for node_idx in top_nodes:
        mask = assignments == node_idx
        counts = np.bincount(labels[mask], minlength=len(class_names))
        total = counts.sum()
        fracs = counts / max(total, 1)
        top3 = np.argsort(-fracs)[:3]
        desc = ", ".join(
            f"{class_names[c]}={fracs[c]:.0%}" for c in top3)
        print(f"  node {node_idx:2d} (n={total:4d}): {desc}")
    print()


# =====================================================================
#  Step 3: Geographic Barabási–Albert graph (real geodesic distance)
# =====================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine (great-circle) distance in km between two (lat, lon)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_geographic_ba_graph(
    centers: np.ndarray,
    m: int = 2,
    distance_scale_km: float = 500.0,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Barabási–Albert graph with distance-decayed preferential attachment.

    Attachment score:   s(j) = (degree(j) + 1) * exp(-dist(i, j) / scale)
    Gives scale-free degree distributions with geographic locality.
    """
    rng = np.random.default_rng(seed)
    n = len(centers)
    adj = {i: set() for i in range(n)}
    degree = np.zeros(n, dtype=int)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(
                centers[i, 0], centers[i, 1],
                centers[j, 0], centers[j, 1])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Seed: complete graph on first m+1 nodes.
    for i in range(min(m + 1, n)):
        for j in range(i + 1, min(m + 1, n)):
            adj[i].add(j)
            adj[j].add(i)
            degree[i] += 1
            degree[j] += 1

    for new_node in range(m + 1, n):
        existing = list(range(new_node))
        scores = np.array([
            (degree[j] + 1) * np.exp(-dist_matrix[new_node, j] / distance_scale_km)
            for j in existing
        ])
        scores /= scores.sum()

        chosen = set()
        attempts = 0
        while len(chosen) < min(m, len(existing)) and attempts < 1000:
            j = rng.choice(existing, p=scores)
            if j not in chosen:
                chosen.add(j)
            attempts += 1

        for j in chosen:
            adj[new_node].add(int(j))
            adj[int(j)].add(new_node)
            degree[new_node] += 1
            degree[int(j)] += 1

    # Normalise to plain Python ints so torch.save AND json.dump are happy.
    adj_list = {int(i): sorted(int(j) for j in nbrs) for i, nbrs in adj.items()}

    degrees = [len(v) for v in adj_list.values()]
    n_edges = sum(degrees) // 2
    print(f"[GRAPH] Geographic BA: n={n} edges={n_edges} m={m} "
          f"scale={distance_scale_km}km")
    print(f"[GRAPH] degree: min={min(degrees)} mean={np.mean(degrees):.1f} "
          f"median={np.median(degrees):.1f} max={max(degrees)}")

    edge_dists = [dist_matrix[i, j] for i, ns in adj_list.items()
                  for j in ns if j > i]
    edge_dists = np.array(edge_dists)
    print(f"[GRAPH] edge distances (km): "
          f"min={edge_dists.min():.0f} mean={edge_dists.mean():.0f} "
          f"median={np.median(edge_dists):.0f} max={edge_dists.max():.0f}")

    return adj_list


# =====================================================================
#  Step 4: Feature extraction
# =====================================================================

def _get_mobilenet_v2_pretrained():
    if MobileNet_V2_Weights is not None:
        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    return mobilenet_v2(pretrained=True)


def _get_efficientnet_b0_pretrained():
    if EfficientNet_B0_Weights is not None:
        return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    return efficientnet_b0(pretrained=True)


class FrozenFeatureExtractor(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        arch = arch.lower()
        self.arch = arch
        if arch == "mobilenet_v2":
            base = _get_mobilenet_v2_pretrained()
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = int(base.last_channel)
        elif arch == "efficientnet_b0":
            base = _get_efficientnet_b0_pretrained()
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = int(base.classifier[-1].in_features)
        else:
            raise ValueError("arch must be mobilenet_v2 or efficientnet_b0")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.pool(f)
        return torch.flatten(f, 1)


class Sentinel2RGBDataset(Dataset):
    """Yields (img, label, index) tuples for the feature extractor.

    2026-04-21 rewrite: reads the 13-band GeoTIFF with rasterio and
    pulls out B04/B03/B02 as RGB for ImageNet-pretrained backbones.
    Applies the same resize/crop/normalize pipeline as the original
    script, only the front-end (image decode) has changed.
    """
    def __init__(self, base_ds: EuroSATWithCoords, input_size: int):
        self.base = base_ds
        # Pipeline operates on CHW float32 tensors in [0, 1] (rasterio
        # path), so we use tensor-compatible transforms.
        self.transform = T.Compose([
            T.Resize(input_size + 32, antialias=True),
            T.CenterCrop(input_size),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, cls_idx, _, _ = self.base.samples[idx]
        with rasterio.open(path) as ds:
            rgb = _read_sentinel2_rgb(ds)  # CHW float32 in [0, 1]
        rgb = self.transform(rgb)
        return rgb, cls_idx, idx


@torch.no_grad()
def extract_features(
    dataset: Dataset,
    model: FrozenFeatureExtractor,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    all_feats, all_labs = [], []
    for imgs, labs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            feats = model.forward_features(imgs)
        all_feats.append(feats.cpu().to(out_dtype))
        all_labs.append(labs.cpu().long())
    return torch.cat(all_feats), torch.cat(all_labs)


# =====================================================================
#  Step 5: Hub assignment for heterogeneous architectures
# =====================================================================

def assign_hub_architectures(
    adj_list: Dict[int, List[int]],
    num_nodes: int,
    efnet_frac: float = 0.10,
) -> Dict[int, str]:
    """Assign EfficientNet to top-degree nodes, MobileNetV2 to rest.

    Mirrors the random_models setup in learned_trust.py.
    """
    degrees = {i: len(nbrs) for i, nbrs in adj_list.items()}
    sorted_nodes = sorted(degrees, key=lambda x: -degrees[x])
    n_efnet = max(1, int(num_nodes * efnet_frac))
    arch_map = {}
    for i, node in enumerate(sorted_nodes):
        arch_map[int(node)] = "efficientnet_b0" if i < n_efnet else "mobilenet_v2"
    n_mn = sum(1 for v in arch_map.values() if v == "mobilenet_v2")
    n_ef = sum(1 for v in arch_map.values() if v == "efficientnet_b0")
    print(f"[ARCH] MobileNetV2={n_mn}  EfficientNet-B0={n_ef}")
    return arch_map


# =====================================================================
#  Step 6: Map plotting on a real Europe basemap
# =====================================================================

# 2026-04-21: reference cities used for orientation on the basemap.
# Sparse on purpose — 8 labels are enough to pin down regions without
# crowding the data overlay.
_REFERENCE_CITIES = [
    ("London",    51.51,  -0.13),
    ("Paris",     48.86,   2.35),
    ("Madrid",    40.42,  -3.70),
    ("Rome",      41.90,  12.50),
    ("Berlin",    52.52,  13.40),
    ("Warsaw",    52.23,  21.01),
    ("Stockholm", 59.33,  18.07),
    ("Athens",    37.98,  23.73),
]


def plot_geographic_clusters(
    coords: np.ndarray,
    node_assign: np.ndarray,
    node_centers: np.ndarray,
    adj_list: Dict[int, List[int]],
    labels: np.ndarray,
    class_names: List[str],
    out_path: str,
    *,
    seed: int = 0,
    point_sample: int = 8000,
    show_cities: bool = True,
) -> List[str]:
    """Save three SEPARATE PNGs — one per panel — showing:
        1. Geographic clusters + BA graph overlay
        2. Points colored by EuroSAT class
        3. BA graph topology (centres sized by cluster population,
           coloured by node degree)

    2026-04-22 refactor: previously produced one combined 22×8 figure
    with all three panels side by side. The user asked for individual
    plots so panels can be picked and placed separately (e.g. into a
    paper where each figure gets its own caption). Each panel now owns
    its own figure at ~9×8 inches so it reads well on its own.

    2026-04-22 (followup): panel titles were simplified per user
    request — just "Geographic Clusters", "Data colored by EuroSAT
    class", and "EuroSAT Network Topology" — and the metadata suptitle
    (image/node/edge counts, coord source, basemap backend) was
    removed from each panel. All of that information is still in the
    console log and in the saved `.pt` cache's `meta` field.

    `out_path` is the filename stem. The actual files written are:
        <stem>_clusters.png       (panel 1)
        <stem>_classes.png        (panel 2)
        <stem>_topology.png       (panel 3)
    where <stem> is `out_path` with any trailing .png stripped.

    Uses cartopy's Lambert Conformal Conic for a real Europe basemap
    with coastlines, country borders, and land/ocean shading. Falls
    back to a plain scatter if cartopy isn't installed.

    Returns the list of PNG paths actually written (possibly empty if
    matplotlib is unavailable).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("[MAP] matplotlib not available — skipping map plots.")
        return []

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except Exception as e:
        HAS_CARTOPY = False
        print(f"[MAP] cartopy not available ({e.__class__.__name__}) — "
              "falling back to plain scatter. Install with "
              "`pip install cartopy` or `conda install -c conda-forge cartopy` "
              "to get the real Europe basemap.")

    num_nodes = int(node_centers.shape[0])
    num_classes = len(class_names)
    n_total = coords.shape[0]
    n_edges = sum(len(v) for v in adj_list.values()) // 2

    # All coords are real (enforced upstream); no NaN filtering needed,
    # but keep a defensive assert rather than silently dropping data.
    assert np.isfinite(coords).all(), "Non-finite coordinates reached plot_geographic_clusters — upstream invariant violated."

    # Subsample for legibility.
    if n_total > point_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=point_sample, replace=False)
        coords_s = coords[idx]
        na_s = node_assign[idx]
        lb_s = labels[idx]
    else:
        coords_s, na_s, lb_s = coords, node_assign, labels

    # 2026-04-22: matplotlib 3.7+ deprecates cm.get_cmap in favour of
    # matplotlib.colormaps[name]; it'll be removed in 3.11. Use the new
    # API but keep a fallback for older installs (<3.7) where the new
    # accessor doesn't exist.
    try:
        import matplotlib as _mpl
        _cmaps = _mpl.colormaps
        cluster_cmap = _cmaps["gist_ncar"].resampled(max(num_nodes, 8))
        class_cmap = _cmaps["tab10"].resampled(max(num_classes, 10))
    except (AttributeError, KeyError):
        cluster_cmap = cm.get_cmap("gist_ncar", max(num_nodes, 8))
        class_cmap = cm.get_cmap("tab10", max(num_classes, 10))
    cluster_colors = cluster_cmap(np.arange(num_nodes))
    class_colors = class_cmap(np.arange(num_classes))

    # Compute the map extent from the real data with a small margin.
    lat_min, lat_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    lon_min, lon_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    margin_lat = max(1.0, 0.1 * (lat_max - lat_min))
    margin_lon = max(2.0, 0.1 * (lon_max - lon_min))
    extent = [lon_min - margin_lon, lon_max + margin_lon,
              lat_min - margin_lat, lat_max + margin_lat]

    # ------------------------------------------------------------------
    # 2026-04-22: factored the projection / basemap / city-label
    # plumbing into small closures so each of the three panels can
    # create its own figure without code duplication. Previously this
    # logic was inline inside one big function with three ax=axes[i]
    # blocks; splitting into per-figure functions makes it much easier
    # to see what's shared and what's unique per panel.
    # ------------------------------------------------------------------

    def _new_figure():
        """Create a fresh figure + single GeoAxes (or plain Axes) sized
        for a standalone panel (~9×8 in; bigger than the 22×8/3 slot
        each panel used to get inside the combined figure)."""
        if HAS_CARTOPY:
            projection = ccrs.LambertConformal(
                central_longitude=(lon_min + lon_max) / 2,
                central_latitude=(lat_min + lat_max) / 2,
                standard_parallels=(40.0, 60.0),
            )
            fig, ax = plt.subplots(
                figsize=(9, 8),
                subplot_kw=dict(projection=projection))
            data_crs = ccrs.PlateCarree()
            tkw = dict(transform=data_crs)
            return fig, ax, data_crs, tkw
        else:
            fig, ax = plt.subplots(figsize=(9, 8))
            return fig, ax, None, {}

    def _setup_basemap(ax, title, data_crs):
        if HAS_CARTOPY:
            ax.set_extent(extent, crs=data_crs)
            ax.add_feature(cfeature.OCEAN, facecolor="#d6e7f2", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f5f2e4", zorder=0)
            ax.add_feature(cfeature.LAKES, facecolor="#d6e7f2",
                           edgecolor="none", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6,
                           edgecolor="#444444", zorder=2)
            ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.4,
                           edgecolor="#888888", zorder=2)
            gl = ax.gridlines(draw_labels=True, alpha=0.35, linestyle=":",
                              linewidth=0.5, color="#666666")
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}
            ax.set_title(title, pad=10, fontsize=12)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.grid(alpha=0.25, linestyle=":")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(title, fontsize=12)

    def _annotate_cities(ax, tkw):
        if not show_cities:
            return
        for name, lat, lon in _REFERENCE_CITIES:
            if not (extent[0] <= lon <= extent[1]
                    and extent[2] <= lat <= extent[3]):
                continue
            ax.plot(lon, lat, marker="o", markersize=3,
                    color="#222222", markeredgecolor="white",
                    markeredgewidth=0.6, zorder=6, **tkw)
            ax.annotate(
                name, xy=(lon, lat), xytext=(4, 4),
                textcoords="offset points",
                fontsize=7, color="#222222", zorder=7,
                **({"xycoords": tkw["transform"]._as_mpl_transform(ax)}
                   if HAS_CARTOPY else {}))

    # 2026-04-22: resolve the output stem. Previously this function
    # accepted a single out_path and wrote one file; now it writes
    # three and needs a stem for suffixing. We accept either a `.png`
    # path (strip the extension and use the remainder as stem) or a
    # bare stem.
    stem = out_path
    if stem.lower().endswith(".png"):
        stem = stem[:-4]
    # Drop a trailing "_map" if present, since the old code appended it
    # and it would now be redundant with the per-panel suffix.
    if stem.endswith("_map"):
        stem = stem[:-4]
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)

    written: List[str] = []

    # ------------------------------------------------------------------
    # Panel 1: clusters + BA edges
    # ------------------------------------------------------------------
    fig, ax, data_crs, tkw = _new_figure()
    # 2026-04-22: title is now just "Geographic Clusters" per user
    # request — node/edge counts were redundant (also appear in the
    # console log) and the metadata suptitle was noise when these
    # panels are cropped into papers / slides.
    _setup_basemap(ax, "Geographic Clusters", data_crs)
    point_colors = cluster_colors[na_s]
    ax.scatter(coords_s[:, 1], coords_s[:, 0], c=point_colors,
               s=4, alpha=0.5, linewidths=0, zorder=3, **tkw)
    for i, nbrs in adj_list.items():
        for j in nbrs:
            if j > i:
                ax.plot([node_centers[i, 1], node_centers[j, 1]],
                        [node_centers[i, 0], node_centers[j, 0]],
                        color="black", alpha=0.5, linewidth=0.8,
                        zorder=4, **tkw)
    ax.scatter(node_centers[:, 1], node_centers[:, 0],
               c="black", s=65, marker="x", linewidths=1.8,
               zorder=5, **tkw)
    _annotate_cities(ax, tkw)
    fig.tight_layout()
    p1 = f"{stem}_clusters.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p1)
    print(f"  Map [clusters]: {p1}")

    # ------------------------------------------------------------------
    # Panel 2: points coloured by EuroSAT class
    # ------------------------------------------------------------------
    fig, ax, data_crs, tkw = _new_figure()
    _setup_basemap(ax, "Data colored by EuroSAT class", data_crs)
    for c in range(num_classes):
        mask = lb_s == c
        if not mask.any():
            continue
        ax.scatter(coords_s[mask, 1], coords_s[mask, 0],
                   c=[class_colors[c]], s=4, alpha=0.6,
                   linewidths=0, label=class_names[c],
                   zorder=3, **tkw)
    _annotate_cities(ax, tkw)
    # 2026-04-23: 10-class legend was single-column, making the rendered
    # panel taller than the topology panel (no legend) and covering part
    # of the British Isles. Use ncol=2 to halve vertical extent; shrink
    # font and tighten spacing so the legend fits in the upper-right
    # corner without obscuring land-cover data underneath.
    leg = ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=7,
        markerscale=2,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
        handletextpad=0.4,
        handlelength=1.2,
    )
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    # 2026-04-22: no metadata suptitle per user request — axis title
    # already reads "Data colored by EuroSAT class".
    fig.tight_layout()
    p2 = f"{stem}_classes.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p2)
    print(f"  Map [classes]:  {p2}")

    # ------------------------------------------------------------------
    # Panel 3: BA graph topology
    # ------------------------------------------------------------------
    fig, ax, data_crs, tkw = _new_figure()
    # 2026-04-22: title is now just "EuroSAT Network Topology" per
    # user request. The old "marker size ∝ cluster population" hint
    # was redundant — the colorbar labels "Node degree" already and
    # the visual encoding is obvious from the plot.
    _setup_basemap(ax, "EuroSAT Network Topology", data_crs)
    sizes = np.bincount(node_assign, minlength=num_nodes).astype(float)
    marker_sizes = 40.0 + 260.0 * (sizes / max(sizes.max(), 1))
    degrees = np.array([len(adj_list[i]) for i in range(num_nodes)])
    for i, nbrs in adj_list.items():
        for j in nbrs:
            if j > i:
                ax.plot([node_centers[i, 1], node_centers[j, 1]],
                        [node_centers[i, 0], node_centers[j, 0]],
                        color="#333333", alpha=0.55, linewidth=1.0,
                        zorder=3, **tkw)
    sc = ax.scatter(node_centers[:, 1], node_centers[:, 0],
                    c=degrees, s=marker_sizes, cmap="viridis",
                    edgecolors="black", linewidths=0.7,
                    zorder=4, **tkw)
    _annotate_cities(ax, tkw)
    # 2026-04-23: put the colorbar in a SEPARATE Axes added via
    # fig.add_axes() so the map Axes itself is never resized. An inline
    # `fig.colorbar(sc, ax=ax, ...)` always takes horizontal space away
    # from the map (even with small fraction=), which makes the map
    # in this panel narrower than the map in panel (a) (which has no
    # colorbar). A separate cbar Axes outside the map's bounds lets the
    # map match panel (a)'s size exactly. Panel (b)'s overall PNG is
    # slightly wider as a result, which is the intended trade-off.
    #
    # Implementation note: matplotlib's tight_layout() doesn't know
    # about Axes created via fig.add_axes(), so we run it FIRST (to let
    # it place the map at its natural size), then read back the map's
    # position, then add the cbar Axes to the right of it. A second
    # tight_layout call would undo this, so we don't call it again —
    # bbox_inches="tight" on savefig handles the external-cbar cropping.
    fig.tight_layout()
    map_bbox = ax.get_position()
    cb_width_figfrac = 0.018   # colorbar width as a fraction of figure width
    cb_gap_figfrac = 0.012     # horizontal gap between map-right and cbar-left
    cb_height_shrink = 0.65    # cbar is shorter than the map (looks balanced)
    cb_height = map_bbox.height * cb_height_shrink
    cb_bottom = map_bbox.y0 + (map_bbox.height - cb_height) / 2
    cb_left = map_bbox.x1 + cb_gap_figfrac
    cax = fig.add_axes([cb_left, cb_bottom, cb_width_figfrac, cb_height])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Node degree", fontsize=9)
    # 2026-04-22: no metadata suptitle per user request.
    # 2026-04-23: NOT calling fig.tight_layout() here — it would reflow
    # the figure and undo the external-colorbar placement above.
    # bbox_inches="tight" on savefig crops whitespace correctly.
    p3 = f"{stem}_topology.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p3)
    print(f"  Map [topology]: {p3}")

    return written


# =====================================================================
#  Main
# =====================================================================

def _resolve_out_path(args) -> str:
    """Return the absolute .pt path we would save to, or load from when
    --use_cached_features is set.

    2026-04-22: extracted from main() so the normal-save path and the
    cache-load path agree on defaults. Without this, accidentally
    changing the default in one place and not the other would silently
    write a new .pt and then fail to find the cached one.
    """
    out_path = args.out_path.strip()
    if not out_path:
        cache_dir = os.path.join(
            os.path.dirname(args.data_root.rstrip("/")),
            "feature_cache")
        out_path = os.path.join(
            cache_dir,
            f"eurosat_geo_{args.arch}_{args.num_nodes}nodes.pt")
    return os.path.abspath(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Precompute EuroSAT features with real Sentinel-2 "
                    "geographic metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--arch", type=str, default="mobilenet_v2",
                    choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", type=str, default="./data/eurosat",
                    help="Root directory for EuroSATallBands download")
    ap.add_argument("--num_nodes", type=int, default=50)
    ap.add_argument("--ba_m", type=int, default=2)
    ap.add_argument("--distance_scale_km", type=float, default=500.0)
    ap.add_argument("--min_per_node", type=int, default=50)
    ap.add_argument("--fp32", action="store_true",
                    help="Store features as float32 instead of float16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_path", type=str, default="")
    ap.add_argument("--save_coords_json", action="store_true")
    # 2026-04-21: new flags replacing the removed --synthetic_coords.
    ap.add_argument("--use_torchgeo", action="store_true",
                    help="Use torchgeo.datasets.EuroSAT to download and "
                         "verify EuroSATallBands. Otherwise download "
                         "directly from HuggingFace.")
    ap.add_argument("--skip_md5", action="store_true",
                    help="Skip MD5 verification of the downloaded archive.")
    # 2026-04-22: --force_extract is the escape hatch when a previous
    # extraction was interrupted. Normal runs don't need it — the new
    # per-class verification auto-detects and repairs partial trees.
    ap.add_argument("--force_extract", action="store_true",
                    help="Force re-extraction of EuroSATallBands even "
                         "if class folders already exist. Useful when a "
                         "previous extraction was interrupted.")
    # 2026-04-22: --no_coord_cache bypasses the sidecar _eurosat_coord_cache.json
    # that's written after a successful extraction pass. The cache is
    # normally a huge win (~40 min saved on reruns) but this gives an
    # escape hatch for debugging.
    ap.add_argument("--no_coord_cache", action="store_true",
                    help="Skip the sidecar coord cache and re-read every "
                         "GeoTIFF's CRS. Normally the cache is used after "
                         "a successful run to avoid a second 30+ minute "
                         "extraction pass.")
    ap.add_argument("--allow_missing_coords", action="store_true",
                    help="If some GeoTIFFs fail to produce real coordinates, "
                         "drop them instead of aborting. Default is to abort "
                         "(synthetic substitution is never done).")
    ap.add_argument("--save_map", dest="save_map", action="store_true",
                    default=True, help="Save PNG map of clusters + BA graph "
                    "(default on; requires matplotlib).")
    ap.add_argument("--no_save_map", dest="save_map", action="store_false")
    # 2026-04-22: --use_cached_features lets the user skip Steps 0–5
    # (archive check, coord extraction, clustering, BA graph, feature
    # extraction, arch assignment) by reloading them all from a previous
    # run's .pt. The typical use case: you've already spent an hour
    # computing features last night and now you just want to re-render
    # the map with tweaked titles, or regenerate the coords JSON, or
    # otherwise touch the downstream artifacts without recomputing the
    # features themselves. If --out_path is not passed we use the same
    # default-path rule as --out_path does for saves, so a plain
    # `--use_cached_features` works as long as your last run also used
    # defaults for --arch / --num_nodes / --data_root.
    ap.add_argument("--use_cached_features", action="store_true",
                    help="Skip Steps 0–5 and reload coords / cluster "
                         "assignment / BA graph / features / arch_map from "
                         "an existing .pt produced by a previous run. "
                         "Only the post-save stages (JSON export, map) "
                         "are redone. Looks at --out_path, or the same "
                         "default path a normal save would pick.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dtype = torch.float32 if args.fp32 else torch.float16

    print(f"{'=' * 70}")
    print(f"  EuroSAT Geographic Feature Extraction (REAL coords)")
    print(f"  arch={args.arch}  nodes={args.num_nodes}  ba_m={args.ba_m}")
    print(f"  device={device}  dtype={'fp32' if args.fp32 else 'fp16'}")
    print(f"  source={'torchgeo' if args.use_torchgeo else 'direct download'}")
    print(f"{'=' * 70}\n")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 2026-04-22: --use_cached_features short-circuit.
    #
    # When the user just wants to re-render the map or regenerate the
    # coords JSON from a previously-computed .pt, we reload every
    # intermediate artifact (coords, node_assign, node_centers,
    # adj_list, feats, labs, arch_map) from the file and skip the
    # ~1-hour feature extraction + ~40-minute coord pass entirely.
    # This branch jumps straight to the post-save stages.
    # ------------------------------------------------------------------
    if args.use_cached_features:
        cache_path = _resolve_out_path(args)
        if not os.path.isfile(cache_path):
            print(f"[CACHED] --use_cached_features set but no .pt found at "
                  f"{cache_path}.",
                  file=sys.stderr)
            print("  Either pass --out_path pointing at an existing file, "
                  "or drop --use_cached_features to recompute.",
                  file=sys.stderr)
            sys.exit(4)
        print(f"[CACHED] Reloading everything from {cache_path}",
              flush=True)
        # weights_only=False: the blob contains a Python dict and the
        # adj_list/arch_map entries are plain containers, not tensors,
        # so torch.load's new safer default would refuse it.
        blob = torch.load(cache_path, map_location="cpu",
                          weights_only=False)

        # Sanity check the blob shape so we fail early on an unrelated
        # or truncated file instead of deep inside the map code.
        required = {"feats_train", "labs_train", "coords", "node_assign",
                    "node_centers", "adj_list", "arch_map", "class_names",
                    "meta"}
        missing = required - set(blob.keys())
        if missing:
            raise RuntimeError(
                f"Cached .pt at {cache_path} is missing keys {sorted(missing)} "
                f"— this doesn't look like a full EuroSAT cache. "
                f"Recompute without --use_cached_features.")

        # Convert tensors back to the numpy shapes the rest of the
        # script (clustering output, map plotter, JSON export) expects.
        feats = blob["feats_train"]
        labs = blob["labs_train"].long()
        coords = blob["coords"].numpy().astype(np.float32)
        node_assign = blob["node_assign"].numpy().astype(np.int64)
        node_centers = blob["node_centers"].numpy().astype(np.float32)
        adj_list = blob["adj_list"]
        arch_map = blob["arch_map"]
        labels = labs.numpy().astype(np.int64)

        meta = blob.get("meta", {})
        # Warn — don't error — if the current CLI args don't match what
        # the cache was created with. This matters because --num_nodes,
        # --arch, etc. affect the default out_path lookup and also the
        # map's expected semantics.
        for key, cli_val in (("arch", args.arch),
                             ("num_nodes", args.num_nodes),
                             ("ba_m", args.ba_m)):
            cached_val = meta.get(key)
            if cached_val is not None and cached_val != cli_val:
                print(f"[CACHED] WARNING: --{key}={cli_val} but cached "
                      f"meta.{key}={cached_val}. Using the cached value.",
                      file=sys.stderr)

        num_nodes_effective = int(meta.get("num_nodes", args.num_nodes))
        print(f"[CACHED] Loaded in {time.time() - t0:.1f}s  "
              f"({feats.shape[0]} images, {feats.shape[1]}-dim feats, "
              f"{num_nodes_effective} nodes, "
              f"{sum(len(v) for v in adj_list.values()) // 2} edges)",
              flush=True)

        out_path = cache_path  # reuse so the log / summary line up
    else:
        # 0. Obtain the georeferenced archive
        print("[STEP 0] Obtaining EuroSATallBands (georeferenced Sentinel-2)...")
        tif_root = obtain_eurosat_allbands(
            args.data_root,
            use_torchgeo=args.use_torchgeo,
            verify_md5=not args.skip_md5,
            force_extract=args.force_extract)

    # 1. Load dataset with real coordinates
    if not args.use_cached_features:
        print("\n[STEP 1] Reading real coordinates from GeoTIFFs...")
        eurosat = EuroSATWithCoords(
            tif_root=tif_root,
            require_all_valid=not args.allow_missing_coords,
            use_coord_cache=not args.no_coord_cache)

        coords = np.array([(s[2], s[3]) for s in eurosat.samples],
                          dtype=np.float32)
        labels = np.array([s[1] for s in eurosat.samples], dtype=np.int64)
        print(f"  Total images: {len(eurosat)}")
        # Class distribution
        for c in range(len(EUROSAT_CLASSES)):
            n = int((labels == c).sum())
            print(f"    {EUROSAT_CLASSES[c]:25s}: {n:5d}")
        print()

        # 2. Geographic clustering
        print("[STEP 2] Clustering into geographic regions...")
        node_assign, node_centers = cluster_by_geography(
            coords, args.num_nodes, seed=args.seed,
            min_per_node=args.min_per_node)
        print_cluster_stats(node_assign, labels, EUROSAT_CLASSES, args.num_nodes)

        # 3. BA graph
        print("[STEP 3] Building geographic BA graph...")
        adj_list = build_geographic_ba_graph(
            node_centers, m=args.ba_m,
            distance_scale_km=args.distance_scale_km,
            seed=args.seed)

        # 4. Features
        print(f"\n[STEP 4] Extracting {args.arch} features from B04/B03/B02...")
        ds = Sentinel2RGBDataset(eurosat, input_size=args.input_size)
        model = FrozenFeatureExtractor(args.arch).to(device).eval()
        print(f"  Model: {args.arch}  feat_dim={model.feat_dim}")
        feats, labs = extract_features(
            ds, model, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            out_dtype=out_dtype)
        t1 = time.time()
        print(f"  Features: {tuple(feats.shape)}  elapsed={t1 - t0:.1f}s")

        # 5. Hub architectures
        print("\n[STEP 5] Assigning architectures to nodes...")
        arch_map = assign_hub_architectures(adj_list, args.num_nodes)

    # 6. Save (skipped on cache-load path — the .pt is already on disk)
    # 2026-04-22: use the shared helper so cache-load and fresh-save
    # agree on the default path.
    out_path = _resolve_out_path(args)

    if not args.use_cached_features:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        blob = {
            "feats_train": feats.contiguous(),
            "labs_train": labs.long(),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "node_assign": torch.tensor(node_assign, dtype=torch.long),
            "node_centers": torch.tensor(node_centers, dtype=torch.float32),
            "adj_list": adj_list,
            "arch_map": arch_map,
            "class_names": EUROSAT_CLASSES,
            "meta": {
                "dataset": "eurosat_allbands_geo",
                "arch": args.arch,
                "input_size": args.input_size,
                "num_classes": len(EUROSAT_CLASSES),
                "num_nodes": args.num_nodes,
                "ba_m": args.ba_m,
                "distance_scale_km": args.distance_scale_km,
                "feat_dim": int(feats.shape[1]),
                "n_images": int(feats.shape[0]),
                "dtype": str(out_dtype),
                "seed": args.seed,
                # 2026-04-21 rewrite: coords are ALWAYS real Sentinel-2 now.
                "coords_source": "real_sentinel2_geotiff",
                "archive_md5": EUROSATALLBANDS_MD5,
                "bands_used": ["B04", "B03", "B02"],  # rasterio 1-indexed 4,3,2
                "brightness_div": SENTINEL2_BRIGHTNESS_DIV,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        print(f"\n[SAVE] {out_path}")
        torch.save(blob, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  File size: {size_mb:.1f} MB")
    else:
        # 2026-04-22: cache-load path doesn't re-save. `blob` is
        # already bound from the torch.load above, so the JSON export
        # below can still reference blob["meta"] without changes.
        print(f"\n[CACHED] Skipping [SAVE] — reusing existing {out_path}",
              flush=True)

    # Optional coords JSON
    if args.save_coords_json:
        json_path = out_path.replace(".pt", "_coords.json")
        coord_data = {
            "nodes": [
                {
                    "id": int(i),
                    "lat": float(node_centers[i, 0]),
                    "lon": float(node_centers[i, 1]),
                    "size": int((node_assign == i).sum()),
                    "arch": arch_map[int(i)],
                    "neighbors": adj_list[int(i)],
                    "class_dist": {
                        EUROSAT_CLASSES[c]: int(v)
                        for c, v in enumerate(
                            np.bincount(labels[node_assign == i],
                                        minlength=len(EUROSAT_CLASSES)))
                    },
                }
                for i in range(int(node_centers.shape[0]))
            ],
            "meta": blob["meta"],
        }
        with open(json_path, "w") as f:
            json.dump(coord_data, f, indent=2)
        print(f"  Coordinates JSON: {json_path}")

    # Map
    # 2026-04-22: plotter now writes three separate PNGs (one per
    # panel) instead of a combined 3-panel figure. We pass the .pt
    # path as the stem; the plotter strips .pt and adds per-panel
    # suffixes (_clusters.png, _classes.png, _topology.png).
    if args.save_map:
        map_stem = out_path[:-3] if out_path.endswith(".pt") else out_path
        plot_geographic_clusters(
            coords=coords,
            node_assign=node_assign,
            node_centers=node_centers,
            adj_list=adj_list,
            labels=labels,
            class_names=EUROSAT_CLASSES,
            out_path=map_stem,
            seed=args.seed)

    # Summary
    # 2026-04-22: derive node count from the actual node_centers
    # rather than args.num_nodes, so the cache-load path shows the
    # cached value (which may differ from the current CLI args).
    print(f"\n{'=' * 70}")
    print(f"  DONE{' (from cache)' if args.use_cached_features else ''}")
    print(f"  Images:   {feats.shape[0]}")
    print(f"  Features: {feats.shape[1]}-dim {feats.dtype}")
    print(f"  Nodes:    {int(node_centers.shape[0])}")
    print(f"  Graph:    {sum(len(v) for v in adj_list.values()) // 2} edges")
    print(f"  Coords:   REAL (Sentinel-2 GeoTIFFs)")
    print(f"  Time:     {time.time() - t0:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()