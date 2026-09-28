"""
Template loading utilities for handling large compressed files.
"""
import hashlib
import os
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TemplateLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            package_dir = Path(__file__).parent.parent
            self.cache_dir = package_dir / "constants" / "_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, gz_path: str, cache_name: Optional[str] = None) -> Path:
        """
        Compute the on-disk cache path for a given source .gz file.

        When cache_name is not given, it is derived from the resolved source path (not
        just the filename), so two different source files that happen to share a
        basename (e.g. a real constants file and a test fixture) never collide on the
        same cache entry.
        """
        gz_path = Path(gz_path)
        if cache_name is None:
            base_name = gz_path.name
            if base_name.endswith('.gz'):
                base_name = base_name[:-len('.gz')]
            if base_name.endswith('.json'):
                base_name = base_name[:-len('.json')]
            digest = hashlib.sha1(str(gz_path.resolve()).encode('utf-8')).hexdigest()[:10]
            cache_name = f"{base_name}_{digest}"
        return self.cache_dir / f"{cache_name}.json"

    def load_json_gz(self, gz_path: str, cache_name: Optional[str] = None) -> Dict[str, Any]:
        gz_path = Path(gz_path)

        if not gz_path.exists():
            raise FileNotFoundError(f"Gzipped file not found: {gz_path}")

        cache_path = self.get_cache_path(gz_path, cache_name)

        if not cache_path.exists() or self._is_gz_newer(gz_path, cache_path):
            logger.info(f"Decompressing {gz_path} to {cache_path}")
            self._decompress_json(gz_path, cache_path)
        else:
            logger.debug(f"Using cached version: {cache_path}")

        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _is_gz_newer(self, gz_path: Path, cache_path: Path) -> bool:
        return gz_path.stat().st_mtime > cache_path.stat().st_mtime
    
    def _decompress_json(self, gz_path: Path, cache_path: Path):
        with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
            with open(cache_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
    
    def clear_cache(self):
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_cache_size(self) -> int:
        total_size = 0
        for cache_file in self.cache_dir.glob("*.json"):
            total_size += cache_file.stat().st_size
        return total_size

_template_loader = TemplateLoader()

def load_uspto_lookup(gz_path: Optional[str] = None) -> Dict[str, Any]:
    if gz_path is None:
        package_dir = Path(__file__).parent.parent
        gz_path = package_dir / "constants" / "uspto_template_lookup.json.gz"

    return _template_loader.load_json_gz(gz_path)
