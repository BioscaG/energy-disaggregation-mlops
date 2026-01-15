from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import shutil
import kagglehub

def download_ukdale(target_dir: str | Path = "data/raw") -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UK-DALE dataset...")
    path = Path(kagglehub.dataset_download("abdelmdz/uk-dale"))
    print(f"Dataset downloaded to: {path}")

    for item in path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    return target_dir


@dataclass
class PreprocessConfig:
    building: int = 1
    meter: int = 1  # meter1 = mains in UK-DALE
    physical_quantity: str = "power"
    power_type: str = "apparent"  # you saw ('power','apparent')
    resample_rule: Optional[str] = None  # e.g. "6S", "1min", or None to keep original
    fill_method: str = "ffill"  # "ffill" or "interpolate"
    clip_min: Optional[float] = 0.0
    clip_max: Optional[float] = None
    normalize: bool = True  # z-score using global mean/std computed in preprocess
    window_size: int = 1024  # number of samples per training window
    stride: int = 256        # step between windows
    chunk_windows: int = 2000  # how many windows to store per .npz chunk


class MyDataset(Dataset):
    """UK-DALE pandas-HDFStore dataset -> preprocessed windows for PyTorch."""

    def __init__(
        self,
        data_path: Path,
        preprocessed_folder: Optional[Path] = None,
        window_size: int = 1024,
        stride: int = 256,
        return_timestamps: bool = False,
    ) -> None:
        self.data_path = Path(data_path)

        # If preprocessed_folder is provided, load prepared chunks.
        # Otherwise, this Dataset can still be used only after you call preprocess().
        self.preprocessed_folder = Path(preprocessed_folder) if preprocessed_folder else None

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.return_timestamps = return_timestamps

        self._chunks: List[Path] = []
        self._index: List[Tuple[int, int]] = []  # (chunk_id, window_id_in_chunk)
        self._meta = {}

        if self.preprocessed_folder is not None:
            self._load_index()

    def _load_index(self) -> None:
        if not self.preprocessed_folder.exists():
            raise FileNotFoundError(
                f"Preprocessed folder not found: {self.preprocessed_folder}. "
                f"Run preprocess() first."
            )

        meta_path = self.preprocessed_folder / "meta.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.npz in {self.preprocessed_folder}")

        meta = np.load(meta_path, allow_pickle=True)
        self._meta = {k: meta[k].item() if meta[k].dtype == object else meta[k] for k in meta.files}

        self._chunks = sorted(self.preprocessed_folder.glob("chunk_*.npz"))
        if not self._chunks:
            raise FileNotFoundError(f"No chunk_*.npz found in {self.preprocessed_folder}")

        # Build a flat window index
        self._index.clear()
        for ci, chunk_path in enumerate(self._chunks):
            with np.load(chunk_path) as z:
                n = int(z["x"].shape[0])
            for wi in range(n):
                self._index.append((ci, wi))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        if self.preprocessed_folder is None:
            raise RuntimeError("This Dataset is not pointing to preprocessed data. "
                               "Pass preprocessed_folder or run preprocess().")

        ci, wi = self._index[index]
        chunk_path = self._chunks[ci]

        with np.load(chunk_path) as z:
            x = z["x"][wi]  # shape: (window_size,)
            # optional timestamps for the window
            t = z["t"][wi] if "t" in z.files else None

        # Convert to torch tensors (add channel dim if you want [C, T])
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # [1, T]

        if self.return_timestamps and t is not None:
            # store timestamps as int64 unix ns (or seconds, depending on preprocess)
            t = torch.from_numpy(t.astype(np.int64))
            return x, t

        return x

    @staticmethod
    def _read_meter_series(
        h5_path: Path,
        building: int,
        meter: int,
        physical_quantity: str,
        power_type: str,
    ) -> pd.Series:
        key = f"/building{building}/elec/meter{meter}"
        store = pd.HDFStore(str(h5_path), mode="r")
        try:
            df = store[key]
        finally:
            store.close()

        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError(f"Expected MultiIndex columns at {key}, got {df.columns}")

        col = (physical_quantity, power_type)
        if col not in df.columns:
            raise KeyError(f"Column {col} not found. Available: {list(df.columns)}")

        s = df[col].copy()
        s = s.sort_index()
        return s

    @staticmethod
    def _resample_and_fill(
        s: pd.Series,
        rule: Optional[str],
        fill_method: str,
    ) -> pd.Series:
        if rule is None:
            return s

        # Resample to a fixed frequency
        s2 = s.resample(rule).mean()

        if fill_method == "ffill":
            s2 = s2.ffill()
        elif fill_method == "interpolate":
            s2 = s2.interpolate(limit_direction="both")
        else:
            raise ValueError("fill_method must be 'ffill' or 'interpolate'")

        return s2

    @staticmethod
    def _make_windows(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
        if len(values) < window_size:
            return np.empty((0, window_size), dtype=values.dtype)

        starts = np.arange(0, len(values) - window_size + 1, stride, dtype=np.int64)
        windows = np.stack([values[i:i + window_size] for i in starts], axis=0)
        return windows

    @staticmethod
    def _make_time_windows(times_ns: np.ndarray, window_size: int, stride: int) -> np.ndarray:
        if len(times_ns) < window_size:
            return np.empty((0, window_size), dtype=times_ns.dtype)

        starts = np.arange(0, len(times_ns) - window_size + 1, stride, dtype=np.int64)
        windows = np.stack([times_ns[i:i + window_size] for i in starts], axis=0)
        return windows

    def preprocess(self, output_folder: Path, cfg: Optional[PreprocessConfig] = None) -> None:
        """Read raw .h5, create fixed-length windows, save chunks + meta."""
        cfg = cfg or PreprocessConfig()
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # 1) Load meter series
        s = self._read_meter_series(
            self.data_path,
            building=cfg.building,
            meter=cfg.meter,
            physical_quantity=cfg.physical_quantity,
            power_type=cfg.power_type,
        )

        # 2) Resample/fill if requested
        s = self._resample_and_fill(s, cfg.resample_rule, cfg.fill_method)

        # 3) Clip if requested
        if cfg.clip_min is not None or cfg.clip_max is not None:
            s = s.clip(lower=cfg.clip_min, upper=cfg.clip_max)

        # 4) Convert timestamps to int64 nanoseconds (fast + compact)
        times_ns = s.index.view("int64")  # datetime64[ns] representation

        # 5) Normalize (z-score) using global stats
        values = s.to_numpy(dtype=np.float32)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values) + 1e-8)

        if cfg.normalize:
            values = (values - mean) / std

        # 6) Build windows
        X = self._make_windows(values, cfg.window_size, cfg.stride)
        T = self._make_time_windows(times_ns, cfg.window_size, cfg.stride)

        if X.shape[0] == 0:
            raise RuntimeError(
                f"Not enough data ({len(values)} samples) for window_size={cfg.window_size}"
            )

        # 7) Save in chunks for fast random access
        n_windows = X.shape[0]
        chunk_windows = int(cfg.chunk_windows)

        chunk_id = 0
        for start in range(0, n_windows, chunk_windows):
            end = min(start + chunk_windows, n_windows)
            x_chunk = X[start:end]
            t_chunk = T[start:end]

            np.savez_compressed(
                output_folder / f"chunk_{chunk_id:04d}.npz",
                x=x_chunk.astype(np.float32),
                t=t_chunk.astype(np.int64),
            )
            chunk_id += 1

        # 8) Save metadata (stats + config)
        meta = dict(
            building=cfg.building,
            meter=cfg.meter,
            physical_quantity=cfg.physical_quantity,
            power_type=cfg.power_type,
            resample_rule=cfg.resample_rule,
            fill_method=cfg.fill_method,
            clip_min=cfg.clip_min,
            clip_max=cfg.clip_max,
            normalize=cfg.normalize,
            mean=mean,
            std=std,
            window_size=cfg.window_size,
            stride=cfg.stride,
            n_samples=len(values),
            n_windows=n_windows,
            timezone=str(getattr(s.index, "tz", None)),
        )
        np.savez_compressed(output_folder / "meta.npz", **meta)


# --- CLI wrapper (typer) ---
import typer

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    cfg = PreprocessConfig(
        building=1,
        meter=1,               # mains
        physical_quantity="power",
        power_type="apparent", # matches what you saw
        resample_rule="6S",    # optional; set None to keep original
        window_size=1024,
        stride=256,
        normalize=True,
    )
    dataset.preprocess(output_folder, cfg=cfg)
    print(f"Done. Saved to: {output_folder}")
