"""Shared fixtures: one freshly seeded database per test session (never the dev copy)."""

from pathlib import Path

import pytest

from app.config import Settings
from app.db import SQLiteDatabase
from data.seed import build


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "shop.db"
    build(path)
    return path


@pytest.fixture()
def settings(db_path: Path, tmp_path: Path) -> Settings:
    return Settings(db_path=db_path, checkpoint_db=tmp_path / "ckpt.db", _env_file=None)


@pytest.fixture()
def db(settings: Settings) -> SQLiteDatabase:
    return SQLiteDatabase(settings.db_path, timeout_s=settings.query_timeout_s, max_rows=settings.row_limit)
