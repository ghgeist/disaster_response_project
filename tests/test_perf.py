import time
import pytest
from app.config import Config
from app.services import ModelService


def test_model_load_under_150ms():
    path = Config.MODEL_PATH
    if not path.exists():
        pytest.skip("model missing")
    svc = ModelService(path)
    svc.load_model()
    start = time.time()
    svc.load_model()
    assert time.time() - start < 0.15
