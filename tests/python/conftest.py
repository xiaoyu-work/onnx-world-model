from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def world_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("mobius")
    from tools.export_mobius_test_model import export_test_model

    return export_test_model(tmp_path_factory.mktemp("world-model"))


@pytest.fixture(scope="session")
def pipeline_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("mobius")
    from tools.export_mobius_test_model import export_test_pipeline

    return export_test_pipeline(tmp_path_factory.mktemp("pipeline"))
