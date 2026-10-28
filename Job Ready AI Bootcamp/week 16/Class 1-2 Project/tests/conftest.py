import pytest

from config import get_config
from src.store.db import MarketStore


@pytest.fixture
def store(tmp_path):
    """A fresh SQLite store in a temp dir, auto-loaded from the bundled Offline Snapshot."""
    return MarketStore(tmp_path / "market.db", get_config().data.snapshot_dir)
