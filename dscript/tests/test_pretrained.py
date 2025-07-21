import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from dscript.pretrained import (
    ROOT_URL,
    STATE_DICT_BASENAME,
    VALID_MODELS,
    build_human_1,
    build_human_tt3d,
    build_lm_1,
    get_pretrained,
    get_state_dict,
    get_state_dict_path,
    retry,
)

MODEL_VERSIONS = [
    "human_v1",  # Original D-SCRIPT Model
    "human_v2",  # Topsy-Turvy
    "human_tt3d",  # TT3D
    "lm_v1",  # Bepler & Berger 2019
]


class TestPretrainedHelpers:
    """Test helper functions in pretrained module."""

    def test_get_state_dict_path(self):
        """Test state dict path generation."""
        path = get_state_dict_path("human_v1")
        assert "dscript_human_v1.pt" in path
        assert os.path.isabs(path)

        path2 = get_state_dict_path("lm_v1")
        assert "dscript_lm_v1.pt" in path2

    def test_valid_models_dict(self):
        """Test that VALID_MODELS contains expected models."""
        expected_models = {"human_v1", "human_v2", "human_tt3d", "lm_v1"}
        assert set(VALID_MODELS.keys()) == expected_models

        # Test that all values are callable
        for model_name, builder_func in VALID_MODELS.items():
            assert callable(builder_func)

    def test_state_dict_basename_format(self):
        """Test state dict basename formatting."""
        expected = "dscript_{version}.pt"
        assert STATE_DICT_BASENAME == expected

        # Test formatting
        formatted = STATE_DICT_BASENAME.format(version="test")
        assert formatted == "dscript_test.pt"

    def test_root_url_format(self):
        """Test ROOT_URL is properly formatted."""
        assert ROOT_URL.startswith("http")
        assert "dscript" in ROOT_URL.lower()


class TestRetryDecorator:
    """Test the retry decorator."""

    def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""

        @retry(3)
        def successful_func(version="test"):
            return f"success_{version}"

        result = successful_func()
        assert result == "success_test"

        result = successful_func("custom")
        assert result == "success_custom"

    def test_retry_decorator_with_kwargs(self):
        """Test retry decorator with kwargs."""

        @retry(3)
        def func_with_kwargs(version="default", other_param=None):
            return f"result_{version}_{other_param}"

        result = func_with_kwargs(version="test", other_param="value")
        assert result == "result_test_value"

    @patch("dscript.pretrained.get_state_dict_path")
    @patch("dscript.pretrained.log")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_retry_decorator_eof_error(
        self, mock_remove, mock_exists, mock_log, mock_get_path
    ):
        """Test retry decorator handles EOF errors properly."""
        mock_get_path.return_value = "/fake/path"
        mock_exists.return_value = True

        call_count = 0

        @retry(2)
        def failing_func(version="test"):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("unexpected EOF, expected 4 bytes got 0")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 2
        mock_remove.assert_called_once_with("/fake/path")

    def test_retry_decorator_max_attempts(self):
        """Test retry decorator reaches max attempts."""

        @retry(2)
        def always_failing_func(version="test"):
            raise RuntimeError("unexpected EOF, expected 4 bytes got 0")

        with (
            patch("dscript.pretrained.get_state_dict_path") as mock_path,
            patch("os.path.exists") as mock_exists,
            patch("os.remove"),
        ):
            mock_path.return_value = "/fake/path"
            mock_exists.return_value = True

            with pytest.raises(Exception, match="Failed to download test"):
                always_failing_func()

    def test_retry_decorator_non_eof_error(self):
        """Test retry decorator re-raises non-EOF errors."""

        @retry(2)
        def func_with_other_error(version="test"):
            raise RuntimeError("Some other error")

        with pytest.raises(RuntimeError, match="Some other error"):
            func_with_other_error()


class TestModelBuilders:
    """Test individual model builder functions with real state dictionaries."""

    def test_build_lm_1(self):
        """Test build_lm_1 function with real state dict."""
        state_dict_path = get_state_dict("lm_v1", verbose=False)
        model = build_lm_1(state_dict_path)

        # Check that model is in eval mode
        assert not model.training

        # Check model architecture
        from dscript.models.embedding import SkipLSTM

        assert isinstance(model, SkipLSTM)

    def test_build_human_1(self):
        """Test build_human_1 function with real state dict."""
        state_dict_path = get_state_dict("human_v1", verbose=False)
        model = build_human_1(state_dict_path)

        # Check that model is in eval mode
        assert not model.training

        # Check model architecture
        from dscript.models.interaction import ModelInteraction

        assert isinstance(model, ModelInteraction)

        # Check that the model has the expected components
        assert hasattr(model, "embedding")
        assert hasattr(model, "contact")

    def test_build_human_v2(self):
        """Test build_human_1 function with human_v2 state dict."""
        state_dict_path = get_state_dict("human_v2", verbose=False)
        model = build_human_1(state_dict_path)

        # Check that model is in eval mode
        assert not model.training

        # Check model architecture
        from dscript.models.interaction import ModelInteraction

        assert isinstance(model, ModelInteraction)

        # Check that the model has the expected components
        assert hasattr(model, "embedding")
        assert hasattr(model, "contact")

    def test_build_human_tt3d(self):
        """Test build_human_tt3d function with real state dict."""
        state_dict_path = get_state_dict("human_tt3d", verbose=False)
        model = build_human_tt3d(state_dict_path)

        # Check that model is in eval mode
        assert not model.training

        # Check model architecture
        from dscript.models.interaction import ModelInteraction

        assert isinstance(model, ModelInteraction)

        # Check that the model has the expected components
        assert hasattr(model, "embedding")
        assert hasattr(model, "contact")


class TestGetStateDict:
    """Test get_state_dict function."""

    @patch("os.path.exists")
    def test_get_state_dict_file_exists(self, mock_exists):
        """Test get_state_dict when file already exists."""
        mock_exists.return_value = True

        with patch("dscript.pretrained.get_state_dict_path") as mock_get_path:
            mock_get_path.return_value = "/fake/existing/path"

            result = get_state_dict("human_v1", verbose=False)
            assert result == "/fake/existing/path"

    @patch("os.path.exists")
    @patch("urllib.request.urlopen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("shutil.copyfileobj")
    def test_get_state_dict_download_success(
        self, mock_copy, mock_file, mock_urlopen, mock_exists
    ):
        """Test successful download when file doesn't exist."""
        mock_exists.return_value = False

        with patch("dscript.pretrained.get_state_dict_path") as mock_get_path:
            mock_get_path.return_value = "/fake/download/path"

            result = get_state_dict("human_v1", verbose=True)

            assert result == "/fake/download/path"
            mock_copy.assert_called_once()

    @patch("os.path.exists")
    @patch("urllib.request.urlopen")
    @patch("sys.exit")
    @patch("dscript.pretrained.log")
    def test_get_state_dict_download_failure(
        self, mock_log, mock_exit, mock_urlopen, mock_exists
    ):
        """Test download failure handling."""
        mock_exists.return_value = False
        mock_urlopen.side_effect = Exception("Network error")

        with patch("dscript.pretrained.get_state_dict_path") as mock_get_path:
            mock_get_path.return_value = "/fake/download/path"

            get_state_dict("human_v1", verbose=False)

            mock_log.assert_called()
            mock_exit.assert_called_once_with(1)


class TestGetPretrained:
    """Test get_pretrained function."""

    def test_get_pretrained_invalid_version(self):
        """Test get_pretrained with invalid model version."""
        with pytest.raises(ValueError, match="Model invalid_model does not exist"):
            get_pretrained("invalid_model")

    @patch("dscript.pretrained.get_state_dict")
    @patch("dscript.pretrained.VALID_MODELS")
    def test_get_pretrained_valid_versions(self, mock_valid_models, mock_get_state_dict):
        """Test get_pretrained with valid versions."""

        def mock_builder(path):
            return f"model_from_{path}"

        mock_valid_models.__contains__ = lambda self, key: key == "test_model"
        mock_valid_models.__getitem__ = lambda self, key: mock_builder
        mock_get_state_dict.return_value = "/fake/state/dict"

        result = get_pretrained("test_model")
        assert result == "model_from_/fake/state/dict"
        mock_get_state_dict.assert_called_once_with("test_model")


# Integration tests for downloading and basic functionality
def test_get_state_dict_integration():
    """Integration test for downloading all model state dictionaries."""
    for mv in MODEL_VERSIONS:
        sd = get_state_dict(mv, verbose=True)
        assert Path(sd).exists(), f"Path {sd} was not downloaded or does not exist"


def test_get_pretrained_integration():
    """Integration test for getting all pretrained models."""
    get_pretrained("human_v1")
    get_pretrained("human_v2")
    get_pretrained("human_tt3d")
    get_pretrained("lm_v1")
