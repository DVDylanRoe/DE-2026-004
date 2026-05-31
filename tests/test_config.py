from unittest.mock import mock_open, patch
from config import load_yaml

def test_load_yaml():
    # fake yaml
    # patch open to read fake yaml
    # read fake yaml
    # assert fake yaml attribute is correct
    input_yaml = "uid: 123"
    with patch("builtins.open", mock_open(read_data=input_yaml)):
        config = load_yaml("config.yml")
    assert config["uid"] == 123
