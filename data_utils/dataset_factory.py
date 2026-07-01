"""
Dataset registry and lightweight loaders for multiple point-cloud
classification datasets, exposed through a single consistent CLI.
Default remains ModelNet40.

Supported datasets
-------------------
- ``modelnet40`` / ``modelnet10``: canonical ModelNet loader (unchanged).
- ``omni_object3d``: class-per-folder point clouds/meshes (sound multi-class).
- ``ycb``: flat folder of meshes/point clouds; class inferred from filename stem.
- ``gazebosim``: flat folder of OBJ meshes; class inferred from filename stem.
- ``craniosynostosis``: PLY ``instances_*`` folders (preferred) or HDF5 submodels.

Preprocessing and splitting
---------------------------
* **Parse-once preprocessing** (``--cache_dataset``, default on): every mesh /
  point file is read, sub-sampled to ``num_point`` and normalised once and
  stored in a single contiguous ``float32`` array.  On Linux the array is shared
  copy-on-write with DataLoader workers, avoiding repeated OBJ/PLY text-parsing
  across epochs for the folder datasets.
* **Stratified per-class splits**: every class with >= 2 samples contributes to
  *both* train and test.
* **Singleton handling** (``--min_class_samples``): classes with fewer than the
  threshold number of samples are dropped and labels re-compacted, so the
  diagnostic datasets (ycb / gazebosim / craniosynostosis) do not error on
  classes that cannot be split.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from .ModelNetDataLoader import ModelNetDataLoader, pc_normalize
from .pc_ops import (
    coerce_channels as _coerce_channels,
    sample_points,
    stratified_split as _stratified_split,
    filter_and_relabel as _pc_filter_and_relabel,
)

# Prefer the optional ``plyfile`` package (listed in requirements.txt) for reading
# PLY datasets (OmniObject3D / YCB / Craniosynostosis).  If a vendored copy is
# shipped under a local ``visualizer``/``visualizations`` directory, make it
# importable too.  When neither is available we fall back to the dependency-free
# parser ``_parse_ply_fallback`` below, so PLY reading never hard-fails.
for _vendor_name in ("visualizer", "visualizations"):
    _vendor_dir = Path(__file__).resolve().parents[1] / _vendor_name
    if _vendor_dir.is_dir() and str(_vendor_dir) not in sys.path:
        sys.path.append(str(_vendor_dir))
try:  # type: ignore
    from plyfile import PlyData  # noqa
except Exception:  # pragma: no cover - optional dependency; built-in fallback used
    PlyData = None  # type: ignore

ALLOWED_EXTENSIONS = {".ply", ".obj", ".txt", ".xyz", ".pts"}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class PointCloudFileDataset(Dataset):
    """Lazy point-cloud dataset backed by a list of files and labels.

    Parsed raw clouds are cached in memory (parse-once) when ``cache_dataset``
    is set on ``args``; sampling/normalisation still run per access so each
    epoch sees a fresh random sub-sample.  All returned arrays are copies, so
    downstream in-place augmentation does not modify the cache.
    """

    def __init__(self, files, labels, args, class_names=None) -> None:
        self.files = list(files)
        self.labels = list(labels)
        self.npoints = args.num_point
        self.use_normals = getattr(args, "use_normals", False)
        self.uniform = getattr(args, "use_uniform_sample", False)
        self.class_names = list(class_names) if class_names else None
        self._cache = {} if getattr(args, "cache_dataset", True) else None

    def __len__(self) -> int:
        return len(self.files)

    def _raw(self, index: int) -> np.ndarray:
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        pts = load_point_cloud(self.files[index]).astype(np.float32)
        if self._cache is not None:
            self._cache[index] = pts
        return pts

    def __getitem__(self, index: int):
        pts = self._raw(index).copy()
        pts = sample_points(pts, self.npoints, use_fps=self.uniform)
        pts = _coerce_channels(pts, self.use_normals)
        pts[:, 0:3] = pc_normalize(pts[:, 0:3])
        return pts.astype(np.float32), int(self.labels[index])


class PointCloudArrayDataset(Dataset):
    """In-memory point-cloud dataset.

    If ``preprocessed=True`` the stored array is already sampled+normalised and
    is returned (as a copy) directly; otherwise sampling/normalisation are
    applied on access.  A copy is returned so in-place normalisation never
    mutates the shared array.
    """

    def __init__(self, points: np.ndarray, labels: np.ndarray, args,
                 class_names=None, preprocessed: bool = False) -> None:
        self.points = points
        self.labels = np.asarray(labels)
        self.npoints = args.num_point
        self.use_normals = getattr(args, "use_normals", False)
        self.uniform = getattr(args, "use_uniform_sample", False)
        self.class_names = list(class_names) if class_names else None
        self.preprocessed = preprocessed

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int):
        pts = np.array(self.points[index], dtype=np.float32, copy=True)
        if not self.preprocessed:
            pts = sample_points(pts, self.npoints, use_fps=self.uniform)
            pts = _coerce_channels(pts, self.use_normals)
            pts[:, 0:3] = pc_normalize(pts[:, 0:3])
        return pts.astype(np.float32), int(self.labels[index])


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_classification_datasets(args):
    """Return ``(train_ds, test_ds, num_classes, class_names)`` based on ``args``."""

    dataset = getattr(args, "dataset", "modelnet40").lower()
    data_root = Path(getattr(args, "data_root", "data")).expanduser()
    train_split = float(getattr(args, "train_split", 0.8))
    seed = int(getattr(args, "dataset_seed", 0))

    if dataset in {"modelnet", "modelnet40", "modelnet_40"}:
        args.num_category = 40
        root = data_root / "modelnet40_normal_resampled"
        process_flag = getattr(args, "process_data", False)
        train_ds = ModelNetDataLoader(root=str(root), args=args, split="train", process_data=process_flag)
        test_ds = ModelNetDataLoader(root=str(root), args=args, split="test", process_data=process_flag)
        return train_ds, test_ds, 40, _read_modelnet_classes(root, 40)

    if dataset in {"modelnet10", "modelnet_10"}:
        args.num_category = 10
        root = data_root / "modelnet40_normal_resampled"
        process_flag = getattr(args, "process_data", False)
        train_ds = ModelNetDataLoader(root=str(root), args=args, split="train", process_data=process_flag)
        test_ds = ModelNetDataLoader(root=str(root), args=args, split="test", process_data=process_flag)
        return train_ds, test_ds, 10, _read_modelnet_classes(root, 10)

    if dataset == "omni_object3d":
        records, class_names = _scan_folder_dataset(data_root / "omni_object3d")
    elif dataset == "ycb":
        records, class_names = _scan_flat_dataset(data_root / "ycb")
    elif dataset == "gazebosim":
        records, class_names = _scan_flat_dataset(data_root / "gazebosim")
    elif dataset == "craniosynostosis":
        return _build_cranio_dataset(data_root / "craniosynostosis", train_split, seed, args)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    train, test, kept = _finalize_file_records(records, class_names, train_split, seed, args)
    args.num_category = len(kept)
    return train, test, len(kept), kept


# ---------------------------------------------------------------------------
# Scanners (return (file, label) records + class names)
# ---------------------------------------------------------------------------


def _read_modelnet_classes(root: Path, num_category: int) -> List[str]:
    catfile = root / ("modelnet10_shape_names.txt" if num_category == 10 else "modelnet40_shape_names.txt")
    if catfile.exists():
        return [line.rstrip() for line in catfile.read_text().splitlines() if line.strip()]
    return []


def _scan_folder_dataset(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """``root/<class_name>/*.{ply,obj,txt}`` layout."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    records: list = []
    class_names: list = []
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = _collect_files(cls_dir)
        if not files:
            continue
        cls_idx = len(class_names)
        class_names.append(cls_dir.name)
        records.extend((f, cls_idx) for f in files)
    if not records:
        raise RuntimeError(f"No point-cloud/mesh files found under {root}")
    return records, class_names


def _scan_flat_dataset(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Flat files; class inferred from filename stem (trailing digits stripped)."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    files = _collect_files(root)
    if not files:
        raise RuntimeError(f"No point-cloud/mesh files found under {root}")
    class_names = sorted({_normalize_class_name(p.stem) for p in files})
    class_map = {name: idx for idx, name in enumerate(class_names)}
    records = [(f, class_map[_normalize_class_name(f.stem)]) for f in files]
    return records, class_names


def _build_cranio_dataset(root: Path, train_split: float, seed: int, args):
    """Craniosynostosis: prefer non-empty PLY ``instances_*`` folders, else HDF5.

    ``instances_full`` is the pooled "all classes combined" set (per the dataset
    README), so it is treated as a class folder only when it is the sole source
    of data; otherwise it is skipped to avoid duplicating the other classes'
    instances.  The per-pathology layout has four classes: control, coronal,
    metopic, sagittal.
    """
    instance_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("instances_")])
    per_class = [p for p in instance_dirs if p.name != "instances_full"]
    if any(_collect_files(p) for p in per_class):
        instance_dirs = per_class
    records: list = []
    class_names: list = []
    for cls_dir in instance_dirs:
        files = _collect_files(cls_dir)
        if not files:
            continue
        cls_idx = len(class_names)
        class_names.append(cls_dir.name.replace("instances_", "") or cls_dir.name)
        records.extend((f, cls_idx) for f in files)
    if records:
        train, test, kept = _finalize_file_records(records, class_names, train_split, seed, args)
        args.num_category = len(kept)
        return train, test, len(kept), kept
    train, test, kept = _build_h5_dataset(root, train_split, seed, args)
    args.num_category = len(kept)
    return train, test, len(kept), kept


def _build_h5_dataset(root: Path, train_split: float, seed: int, args):
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required for HDF5 datasets; add it to your environment") from exc

    h5_files = sorted(root.glob("submodel_*.h5"))
    if not h5_files:
        raise RuntimeError(f"No HDF5 files matching 'submodel_*.h5' found in {root}")

    data_list, label_list, class_names = [], [], []
    for cls_idx, h5_path in enumerate(h5_files):
        class_names.append(h5_path.stem.replace("submodel_", ""))
        with h5py.File(h5_path, "r") as f:
            points = _load_h5_points(f)
            data_list.append(points)
            label_list.append(np.full((points.shape[0],), cls_idx, dtype=np.int64))

    all_points = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    return _finalize_array(all_points, all_labels, class_names, train_split, seed, args)


# ---------------------------------------------------------------------------
# Finalisation: filter singletons -> stratified split -> (optionally) precompute
# ---------------------------------------------------------------------------


def _finalize_file_records(records, class_names, train_split, seed, args):
    """Filter rare classes, re-label, stratified split, optional parse-once cache."""
    labels = np.array([r[1] for r in records], dtype=np.int64)
    keep_records, kept_names, klabels = _filter_and_relabel(records, labels, class_names, args)

    train_idx, test_idx = _stratified_split(klabels, train_split, seed)

    if getattr(args, "cache_dataset", True):
        files = [keep_records[i][0] for i in range(len(keep_records))]
        pts_all = _precompute_points(files, args)
        train = PointCloudArrayDataset(pts_all[train_idx], klabels[train_idx], args, kept_names, preprocessed=True)
        test = PointCloudArrayDataset(pts_all[test_idx], klabels[test_idx], args, kept_names, preprocessed=True)
    else:
        train = PointCloudFileDataset([keep_records[i][0] for i in train_idx],
                                      [klabels[i] for i in train_idx], args, kept_names)
        test = PointCloudFileDataset([keep_records[i][0] for i in test_idx],
                                     [klabels[i] for i in test_idx], args, kept_names)
    return train, test, kept_names


def _finalize_array(points, labels, class_names, train_split, seed, args):
    records = list(range(len(points)))
    keep_idx, kept_names, klabels = _filter_and_relabel(
        [(i, int(labels[i])) for i in records], np.asarray(labels), class_names, args)
    kept_indices = [r[0] for r in keep_idx]
    pts = np.stack([_sample_and_normalize(points[i], args) for i in kept_indices]).astype(np.float32)
    train_idx, test_idx = _stratified_split(klabels, train_split, seed)
    train = PointCloudArrayDataset(pts[train_idx], klabels[train_idx], args, kept_names, preprocessed=True)
    test = PointCloudArrayDataset(pts[test_idx], klabels[test_idx], args, kept_names, preprocessed=True)
    return train, test, kept_names


def _filter_and_relabel(records, labels, class_names, args):
    """Drop classes with < ``min_class_samples`` members and compact labels."""
    min_k = int(getattr(args, "min_class_samples", 1) or 1)
    _, remap, keep_classes = _pc_filter_and_relabel(labels, len(class_names), min_k)
    kept_records = [(p, remap[l]) for (p, l) in records if l in remap]
    kept_names = [class_names[c] for c in keep_classes]
    klabels = np.array([l for _, l in kept_records], dtype=np.int64)
    dropped = len(class_names) - len(kept_names)
    if dropped:
        print(f"[dataset_factory] dropped {dropped} class(es) with < {min_k} samples; "
              f"{len(kept_names)} classes remain.")
    return kept_records, kept_names, klabels


def _precompute_points(files, args) -> np.ndarray:
    out = np.empty((len(files), args.num_point, 6 if getattr(args, "use_normals", False) else 3),
                   dtype=np.float32)
    for i, path in enumerate(files):
        out[i] = _sample_and_normalize(load_point_cloud(path), args)
    return out


def _sample_and_normalize(raw: np.ndarray, args) -> np.ndarray:
    pts = sample_points(np.asarray(raw, dtype=np.float32).copy(), args.num_point,
                        use_fps=getattr(args, "use_uniform_sample", False))
    pts = _coerce_channels(pts, getattr(args, "use_normals", False))
    pts[:, 0:3] = pc_normalize(pts[:, 0:3])
    return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _collect_files(folder: Path) -> list:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in ALLOWED_EXTENSIONS and p.is_file()]


def _normalize_class_name(stem: str) -> str:
    name = re.sub(r"[_-]?\d+$", "", stem)
    return name or stem


def load_point_cloud(path: Path) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return _load_obj_points(path)
    if suffix == ".ply":
        return _load_ply_points(path)
    return _load_txt_points(path)


def _load_obj_points(path: Path) -> np.ndarray:
    vertices: list = []
    normals: list = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()[1:]
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith("vn "):
                parts = line.strip().split()[1:]
                if len(parts) >= 3:
                    normals.append([float(parts[0]), float(parts[1]), float(parts[2])])
    pts = np.array(vertices, dtype=np.float32)
    if normals and len(normals) == len(vertices):
        pts = np.concatenate([pts, np.array(normals, dtype=np.float32)], axis=1)
    return pts


def _load_ply_points(path: Path) -> np.ndarray:
    """Read vertex coords (and normals when present) from a PLY file.

    Uses the optional ``plyfile`` package when it is importable; otherwise falls
    back to the dependency-free :func:`_parse_ply_fallback` so PLY datasets load
    even when ``plyfile`` is not installed.
    """
    if PlyData is not None:
        plydata = PlyData.read(str(path))
        if "vertex" not in plydata:
            raise RuntimeError(f"PLY file missing vertex element: {path}")
        vertex = plydata["vertex"]
        coords = np.stack([np.asarray(vertex[x]) for x in ["x", "y", "z"]], axis=1)
        names = vertex.data.dtype.names or ()
        if {"nx", "ny", "nz"}.issubset(names):
            normals = np.stack([np.asarray(vertex[x]) for x in ["nx", "ny", "nz"]], axis=1)
            return np.concatenate([coords, normals], axis=1).astype(np.float32)
        return coords.astype(np.float32)
    return _parse_ply_fallback(path)


# PLY scalar type names -> numpy dtype codes (byte order added at read time).
_PLY_TYPE_TO_NUMPY = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def _parse_ply_header(fh):
    """Parse a PLY header from a binary file handle positioned at the start.

    Returns ``(fmt, elements)`` where ``fmt`` is the format string (e.g.
    ``ascii``/``binary_little_endian``) and ``elements`` is an ordered list of
    ``(name, count, props)``.  Each entry of ``props`` is either
    ``("scalar", prop_name, type_str)`` or ``("list", prop_name, count_type, item_type)``.
    """
    magic = fh.readline().strip()
    if magic not in (b"ply", b"PLY"):
        raise RuntimeError("not a PLY file (missing 'ply' magic)")
    fmt = None
    elements = []  # list of [name, count, props]
    while True:
        raw = fh.readline()
        if not raw:
            raise RuntimeError("unexpected EOF while reading PLY header")
        tokens = raw.split()
        if not tokens:
            continue
        key = tokens[0]
        if key == b"format":
            fmt = tokens[1].decode("ascii", "ignore")
        elif key == b"element":
            elements.append([tokens[1].decode("ascii", "ignore"), int(tokens[2]), []])
        elif key == b"property":
            if not elements:
                continue
            if tokens[1] == b"list":
                # property list <count_type> <item_type> <name>
                elements[-1][2].append(("list", tokens[4].decode("ascii", "ignore"),
                                        tokens[2].decode("ascii", "ignore"),
                                        tokens[3].decode("ascii", "ignore")))
            else:
                # property <type> <name>
                elements[-1][2].append(("scalar", tokens[2].decode("ascii", "ignore"),
                                        tokens[1].decode("ascii", "ignore")))
        elif key == b"end_header":
            break
    return fmt, elements


def _parse_ply_fallback(path: Path) -> np.ndarray:
    """Dependency-free PLY reader (ASCII + binary little/big-endian).

    Returns the vertex ``x, y, z`` coordinates as an ``(N, 3)`` float32 array, or
    ``(N, 6)`` when per-vertex normals (``nx, ny, nz``) are present.  Only the
    ``vertex`` element is needed, so other elements (e.g. faces) are skipped.
    """
    with open(path, "rb") as fh:
        fmt, elements = _parse_ply_header(fh)
        if fmt is None:
            raise RuntimeError(f"PLY file missing format line: {path}")

        vertex = next((el for el in elements if el[0] == "vertex"), None)
        if vertex is None:
            raise RuntimeError(f"PLY file missing vertex element: {path}")
        _, vcount, vprops = vertex
        scalar_names = [p[1] for p in vprops if p[0] == "scalar"]
        if not {"x", "y", "z"}.issubset(scalar_names):
            raise RuntimeError(f"PLY vertex element missing x/y/z properties: {path}")
        has_normals = {"nx", "ny", "nz"}.issubset(scalar_names)
        wanted = ["x", "y", "z"] + (["nx", "ny", "nz"] if has_normals else [])

        if fmt.startswith("ascii"):
            return _parse_ply_ascii(fh, elements, vertex, wanted)
        byte_order = "<" if "little" in fmt else ">"
        return _parse_ply_binary(fh, elements, vertex, wanted, byte_order)


def _parse_ply_ascii(fh, elements, vertex, wanted) -> np.ndarray:
    """Read the vertex rows of an ASCII PLY, skipping any preceding elements."""
    _, vcount, vprops = vertex
    col_index = {p[1]: i for i, p in enumerate(vprops)}  # property name -> column
    cols = [col_index[name] for name in wanted]
    rows = []
    for el in elements:
        is_vertex = el is vertex
        for _ in range(el[1]):
            line = fh.readline()
            if not line:
                raise RuntimeError("unexpected EOF while reading ASCII PLY body")
            if is_vertex:
                parts = line.split()
                rows.append([float(parts[c]) for c in cols])
        if is_vertex:
            break  # everything we need has been read
    return np.asarray(rows, dtype=np.float32).reshape(vcount, len(wanted))


def _parse_ply_binary(fh, elements, vertex, wanted, byte_order) -> np.ndarray:
    """Read the vertex rows of a binary PLY, skipping any preceding elements."""
    for el in elements:
        name, count, props = el
        if el is vertex:
            dtype = np.dtype([(p[1], byte_order + _PLY_TYPE_TO_NUMPY[p[2]])
                              for p in props if p[0] == "scalar"])
            buf = fh.read(count * dtype.itemsize)
            if len(buf) < count * dtype.itemsize:
                raise RuntimeError("unexpected EOF while reading binary PLY vertices")
            data = np.frombuffer(buf, dtype=dtype, count=count)
            return np.stack([data[name].astype(np.float32) for name in wanted], axis=1)
        # Skip a non-vertex element to reach the vertex data.
        if all(p[0] == "scalar" for p in props):
            skip_dtype = np.dtype([(p[1], byte_order + _PLY_TYPE_TO_NUMPY[p[2]])
                                   for p in props])
            fh.seek(count * skip_dtype.itemsize, 1)
        else:
            # Element contains a variable-length list property: advance row by row.
            for _ in range(count):
                for p in props:
                    if p[0] == "scalar":
                        fh.seek(np.dtype(_PLY_TYPE_TO_NUMPY[p[2]]).itemsize, 1)
                    else:
                        n_raw = fh.read(np.dtype(_PLY_TYPE_TO_NUMPY[p[2]]).itemsize)
                        n = int(np.frombuffer(n_raw, dtype=byte_order + _PLY_TYPE_TO_NUMPY[p[2]])[0])
                        fh.seek(n * np.dtype(_PLY_TYPE_TO_NUMPY[p[3]]).itemsize, 1)
    raise RuntimeError("PLY vertex element not found while scanning binary body")


def _load_txt_points(path: Path) -> np.ndarray:
    try:
        pts = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts


def _load_h5_points(h5_file) -> np.ndarray:
    for key in ("data", "points", "pointcloud", "pc"):
        if key in h5_file:
            return np.asarray(h5_file[key])
    raise RuntimeError("HDF5 file missing expected point dataset (tried data/points/pointcloud/pc)")
# end of dataset_factory.py
