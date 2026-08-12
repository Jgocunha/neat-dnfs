import json
import pandas as pd

_CACHE_DIRNAME = ".viz_cache"


_PARSER_VERSION = 1


def _run_cache_dir(dir_path: Path) -> Path:
    return dir_path / _CACHE_DIRNAME


def _fingerprint_dir(dir_path: Path, glob_pattern: str = "*") -> str:
    """Cheap fingerprint (stat-only, no content hashing) of files matching
    glob_pattern under dir_path: file count, total size, max mtime."""
    count = 0
    total_size = 0
    max_mtime = 0.0
    if dir_path.exists():
        for p in dir_path.glob(glob_pattern):
            if not p.is_file():
                continue
            try:
                s = p.stat()
            except OSError:
                continue
            count += 1
            total_size += s.st_size
            max_mtime = max(max_mtime, s.st_mtime)
    return f"v{_PARSER_VERSION}:{count}:{total_size}:{max_mtime:.3f}"


def _disk_cache_read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _disk_cache_write_json(path: Path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
    except Exception:
        pass


def _disk_cache_read_df(path: Path):
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        pass
    return None


def _disk_cache_write_df(df: pd.DataFrame, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        pass
