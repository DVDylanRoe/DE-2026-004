from unittest.mock import mock_open, patch
from config import load_yaml, resolve_uid


def test_load_yaml():
    input_yaml = "uid: 123"
    with patch("builtins.open", mock_open(read_data=input_yaml)):
        config = load_yaml("config.yml")
    assert config["uid"] == 123


def test_resolve_uid_override():
    args = type("obj", (), {"uid": 555})
    cfg = {"uid": 123}
    uid = resolve_uid(cfg, args)

    assert uid == 555


def test_resolve_uid_default():
    args = type("obj", (), {"uid": None})
    cfg = {"uid": 123}
    uid = resolve_uid(cfg, args)

    assert uid == 123
