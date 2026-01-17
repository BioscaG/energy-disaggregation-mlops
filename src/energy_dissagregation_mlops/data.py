from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger

import shutil
import kagglehub

def download_ukdale(target_dir: str | Path = "data/raw") -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading UK-DALE dataset from Kaggle...")
    path = Path(kagglehub.dataset_download("abdelmdz/uk-dale"))
    logger.success(f"Dataset downloaded to: {path}")

    logger.info(f"Copying dataset to {target_dir}...")
    for item in path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    logger.success(f"Dataset ready at: {target_dir}")
    return target_dir


@dataclass
class PreprocessConfig:
    building: int = 1
    meter_mains: int = 1  # meter1 = mains in UK-DALE
    meter_appliance: int = 2  # meter2+ = appliance to predict
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
    max_samples: Optional[int] = None  # limit total samples for faster preprocessing (None=all)


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

        logger.debug(f"Loading preprocessed data from: {self.preprocessed_folder}")
        meta = np.load(meta_path, allow_pickle=True)
        self._meta = {k: meta[k].item() if meta[k].dtype == object else meta[k] for k in meta.files}

        self._chunks = sorted(self.preprocessed_folder.glob("chunk_*.npz"))
        if not self._chunks:
            raise FileNotFoundError(f"No chunk_*.npz found in {self.preprocessed_folder}")

        logger.debug(f"Found {len(self._chunks)} chunks")

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
            x = z["x"][wi]  # mains power, shape: (window_size,)
            y = z["y"][wi]  # appliance power, shape: (window_size,)
            t = z["t"][wi] if "t" in z.files else None

        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # [1, T]
        y = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)  # [1, T]

        if self.return_timestamps and t is not None:
            t = torch.from_numpy(t.astype(np.int64))
            return x, y, t

        return x, y

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

        logger.info("Starting data preprocessing...")
        logger.info(f"Config: building={cfg.building}, meter_mains={cfg.meter_mains}, meter_appliance={cfg.meter_appliance}")
        logger.info(f"Window: size={cfg.window_size}, stride={cfg.stride}, normalize={cfg.normalize}")

        # 1) Load mains (input) and appliance (target)
        logger.info(f"Loading meter {cfg.meter_mains} (mains) from building {cfg.building}...")
        s_mains = self._read_meter_series(
            self.data_path,
            building=cfg.building,
            meter=cfg.meter_mains,
            physical_quantity=cfg.physical_quantity,
            power_type=cfg.power_type,
        )

        logger.info(f"Loading meter {cfg.meter_appliance} (appliance) from building {cfg.building}...")
        s_appliance = self._read_meter_series(
            self.data_path,
            building=cfg.building,
            meter=cfg.meter_appliance,
            physical_quantity=cfg.physical_quantity,
            power_type=cfg.power_type,
        )

        # 2) Align indices (find common timestamps)
        logger.debug("Aligning timestamps...")
        common_idx = s_mains.index.intersection(s_appliance.index)
        s_mains = s_mains.loc[common_idx]
        s_appliance = s_appliance.loc[common_idx]
        logger.info(f"Common timestamps: {len(common_idx)}")

        # 3) Resample/fill if requested
        if cfg.resample_rule:
            logger.info(f"Resampling to {cfg.resample_rule} with {cfg.fill_method}...")
        s_mains = self._resample_and_fill(s_mains, cfg.resample_rule, cfg.fill_method)
        s_appliance = self._resample_and_fill(s_appliance, cfg.resample_rule, cfg.fill_method)

        # 4) Clip if requested
        if cfg.clip_min is not None or cfg.clip_max is not None:
            logger.debug(f"Clipping values to [{cfg.clip_min}, {cfg.clip_max}]")
            s_mains = s_mains.clip(lower=cfg.clip_min, upper=cfg.clip_max)
            s_appliance = s_appliance.clip(lower=cfg.clip_min, upper=cfg.clip_max)

        # 4b) Limit max samples if requested (for faster testing)
        if cfg.max_samples is not None:
            logger.info(f"Limiting to {cfg.max_samples} samples (max_samples setting)")
            s_mains = s_mains.iloc[:cfg.max_samples]
            s_appliance = s_appliance.iloc[:cfg.max_samples]

        # 5) Convert timestamps to int64 nanoseconds
        times_ns = s_mains.index.view("int64")

        # 6) Normalize (z-score) using global stats per signal
        logger.info("Computing normalization statistics...")
        values_mains = s_mains.to_numpy(dtype=np.float32)
        values_appliance = s_appliance.to_numpy(dtype=np.float32)

        mean_mains = float(np.nanmean(values_mains))
        std_mains = float(np.nanstd(values_mains) + 1e-8)
        mean_appliance = float(np.nanmean(values_appliance))
        std_appliance = float(np.nanstd(values_appliance) + 1e-8)

        logger.debug(f"Mains: mean={mean_mains:.2f}, std={std_mains:.2f}")
        logger.debug(f"Appliance: mean={mean_appliance:.2f}, std={std_appliance:.2f}")

        if cfg.normalize:
            logger.info("Applying z-score normalization...")
            values_mains = (values_mains - mean_mains) / std_mains
            values_appliance = (values_appliance - mean_appliance) / std_appliance

        # 7) Build windows
        logger.info(f"Creating windows (window_size={cfg.window_size}, stride={cfg.stride})...")
        X = self._make_windows(values_mains, cfg.window_size, cfg.stride)
        Y = self._make_windows(values_appliance, cfg.window_size, cfg.stride)
        T = self._make_time_windows(times_ns, cfg.window_size, cfg.stride)

        if X.shape[0] == 0:
            raise RuntimeError(
                f"Not enough data ({len(values_mains)} samples) for window_size={cfg.window_size}"
            )

        logger.info(f"Created {X.shape[0]} windows from {len(values_mains)} samples")

        # 8) Save in chunks for fast random access
        n_windows = X.shape[0]
        chunk_windows = int(cfg.chunk_windows)

        logger.info(f"Saving {n_windows} windows in chunks of {chunk_windows}...")
        chunk_id = 0
        for start in range(0, n_windows, chunk_windows):
            end = min(start + chunk_windows, n_windows)
            x_chunk = X[start:end]
            y_chunk = Y[start:end]
            t_chunk = T[start:end]

            np.savez_compressed(
                output_folder / f"chunk_{chunk_id:04d}.npz",
                x=x_chunk.astype(np.float32),
                y=y_chunk.astype(np.float32),
                t=t_chunk.astype(np.int64),
            )
            logger.debug(f"Saved chunk {chunk_id} ({end-start} windows)")
            chunk_id += 1

        logger.info(f"Saved {chunk_id} chunk files")

        # 9) Save metadata (stats + config)
        logger.info("Saving metadata...")
        meta = dict(
            building=cfg.building,
            meter_mains=cfg.meter_mains,
            meter_appliance=cfg.meter_appliance,
            physical_quantity=cfg.physical_quantity,
            power_type=cfg.power_type,
            resample_rule=cfg.resample_rule,
            fill_method=cfg.fill_method,
            clip_min=cfg.clip_min,
            clip_max=cfg.clip_max,
            normalize=cfg.normalize,
            mean_mains=mean_mains,
            std_mains=std_mains,
            mean_appliance=mean_appliance,
            std_appliance=std_appliance,
            window_size=cfg.window_size,
            stride=cfg.stride,
            n_samples=len(values_mains),
            n_windows=n_windows,
            timezone=str(getattr(s_mains.index, "tz", None)),
        )
        np.savez_compressed(output_folder / "meta.npz", **meta)

        logger.success(f"Preprocessing complete! Saved to: {output_folder}")
        logger.info(f"Total: {n_windows} windows from {len(values_mains)} samples")


# --- CLI wrapper (typer) ---
import typer

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    cfg = PreprocessConfig(
        building=1,
        meter_mains=1,         # mains
        meter_appliance=2,     # appliance to predict
        physical_quantity="power",
        power_type="apparent",
        resample_rule="6S",    # optional; set None to keep original
        window_size=1024,
        stride=256,
        normalize=True,
    )
    dataset.preprocess(output_folder, cfg=cfg)
    print(f"Done. Saved to: {output_folder}")
