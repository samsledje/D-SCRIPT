"""
Tests for utility functions in dscript.utils
"""

import sys
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from dscript.utils import (
    PairedDataset,
    RBF,
    collate_paired_sequences,
    load_hdf5_parallel,
    log,
    parse_device,
    setup_logger,
)


class TestSetupLogger:
    """Tests for setup_logger function"""

    def test_setup_logger_no_file_default_stdout(self, capsys):
        """Test that setup_logger logs to stdout by default"""
        from loguru import logger

        setup_logger()
        logger.info("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_setup_logger_with_file(self, tmp_path):
        """Test that setup_logger writes to a file when provided"""
        from loguru import logger

        log_file = tmp_path / "test.log"
        setup_logger(log_file=str(log_file))
        logger.info("test file message")

        # Read the file content
        content = log_file.read_text()
        assert "test file message" in content

    def test_setup_logger_file_and_stdout(self, tmp_path, capsys):
        """Test that setup_logger can write to both file and stdout"""
        from loguru import logger

        log_file = tmp_path / "test.log"
        setup_logger(log_file=str(log_file), also_stdout=True)
        logger.info("dual output message")

        # Check both stdout and file
        captured = capsys.readouterr()
        assert "dual output message" in captured.out

        content = log_file.read_text()
        assert "dual output message" in content


class TestLog:
    """Tests for log function"""

    def test_log_to_stdout(self, capsys):
        """Test that log writes to stdout by default"""
        log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_log_to_file(self, tmp_path):
        """Test that log writes to a file handle"""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            log("file message", file=f)

        content = log_file.read_text()
        assert "file message" in content

    def test_log_with_print_also(self, tmp_path, capsys):
        """Test that log can write to both file and stdout"""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            log("dual message", file=f, print_also=True)

        captured = capsys.readouterr()
        assert "dual message" in captured.out

        content = log_file.read_text()
        assert "dual message" in content

    def test_log_flushes_file(self, tmp_path):
        """Test that log flushes the file handle"""
        log_file = tmp_path / "test.log"
        mock_file = Mock()
        mock_file.flush = Mock()

        log("flush test", file=mock_file)
        mock_file.flush.assert_called_once()


class TestRBF:
    """Tests for Radial Basis Function"""

    def test_rbf_basic(self):
        """Test basic RBF computation"""
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = RBF(D)

        # Check shape
        assert result.shape == D.shape

        # Diagonal should be 1.0 (distance from point to itself is 0)
        assert np.allclose(result[0, 0], 1.0)
        assert np.allclose(result[1, 1], 1.0)

        # Off-diagonal should be less than 1
        assert result[0, 1] < 1.0
        assert result[1, 0] < 1.0

    def test_rbf_with_custom_sigma(self):
        """Test RBF with custom sigma parameter"""
        D = np.array([[0.0, 2.0], [2.0, 0.0]])
        sigma = 1.0
        result = RBF(D, sigma=sigma)

        # Expected value: exp(-4 / (2 * 1^2)) = exp(-2)
        expected_off_diag = np.exp(-2)
        assert np.allclose(result[0, 1], expected_off_diag)

    def test_rbf_zero_distance(self):
        """Test RBF with zero distance matrix"""
        D = np.zeros((3, 3))
        result = RBF(D)

        # All values should be 1.0
        assert np.allclose(result, 1.0)

    def test_rbf_symmetry(self):
        """Test that RBF produces symmetric output"""
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        result = RBF(D)

        # Result should be symmetric
        assert np.allclose(result, result.T)

    def test_rbf_values_bounded(self):
        """Test that RBF values are in (0, 1]"""
        D = np.random.rand(5, 5) * 10
        result = RBF(D)

        # All values should be in (0, 1]
        assert np.all(result > 0)
        assert np.all(result <= 1)


class TestLoadHDF5Parallel:
    """Tests for load_hdf5_parallel function"""

    @patch("dscript.utils.LoadingPool")
    def test_load_hdf5_parallel_returns_dict(self, mock_loading_pool):
        """Test that load_hdf5_parallel returns a dict by default"""
        # Setup mock
        mock_pool_instance = MagicMock()
        mock_pool_instance.load_once.return_value = ["emb1", "emb2", "emb3"]
        mock_loading_pool.return_value = mock_pool_instance

        # Call function
        keys = ["prot1", "prot2", "prot3"]
        result = load_hdf5_parallel("test.h5", keys, n_jobs=4, return_dict=True)

        # Verify
        mock_loading_pool.assert_called_once_with("test.h5", 4)
        mock_pool_instance.load_once.assert_called_once_with(keys)
        assert isinstance(result, dict)
        assert result == {"prot1": "emb1", "prot2": "emb2", "prot3": "emb3"}

    @patch("dscript.utils.LoadingPool")
    def test_load_hdf5_parallel_returns_list(self, mock_loading_pool):
        """Test that load_hdf5_parallel can return a list"""
        # Setup mock
        mock_pool_instance = MagicMock()
        mock_pool_instance.load_once.return_value = ["emb1", "emb2"]
        mock_loading_pool.return_value = mock_pool_instance

        # Call function
        keys = ["prot1", "prot2"]
        result = load_hdf5_parallel("test.h5", keys, return_dict=False)

        # Verify
        assert isinstance(result, list)
        assert result == ["emb1", "emb2"]

    @patch("dscript.utils.LoadingPool")
    def test_load_hdf5_parallel_default_njobs(self, mock_loading_pool):
        """Test that load_hdf5_parallel uses default n_jobs"""
        mock_pool_instance = MagicMock()
        mock_pool_instance.load_once.return_value = []
        mock_loading_pool.return_value = mock_pool_instance

        load_hdf5_parallel("test.h5", [])
        mock_loading_pool.assert_called_once_with("test.h5", -1)


class TestParseDevice:
    """Tests for parse_device function"""

    def test_parse_device_cpu(self, tmp_path):
        """Test parsing 'cpu' device"""
        log_file = open(tmp_path / "test.log", "w")
        device = parse_device("cpu", log_file)
        assert device == "cpu"
        log_file.close()

    def test_parse_device_cpu_case_insensitive(self, tmp_path):
        """Test that 'CPU' (uppercase) works"""
        log_file = open(tmp_path / "test.log", "w")
        device = parse_device("CPU", log_file)
        assert device == "cpu"
        log_file.close()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    def test_parse_device_all(self, mock_device_count, mock_cuda_available, tmp_path):
        """Test parsing 'all' device"""
        log_file = open(tmp_path / "test.log", "w")
        device = parse_device("all", log_file)
        assert device == -1
        log_file.close()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    def test_parse_device_gpu_index(
        self, mock_device_count, mock_cuda_available, tmp_path
    ):
        """Test parsing GPU index"""
        log_file = open(tmp_path / "test.log", "w")
        device = parse_device("0", log_file)
        assert device == 0

        device = parse_device("2", log_file)
        assert device == 2
        log_file.close()

    def test_parse_device_invalid_string(self, tmp_path):
        """Test that invalid device string causes exit"""
        log_file = open(tmp_path / "test.log", "w")
        with pytest.raises(SystemExit) as exc_info:
            parse_device("invalid", log_file)
        assert exc_info.value.code == 1

    @patch("torch.cuda.is_available", return_value=False)
    def test_parse_device_cuda_not_available(self, mock_cuda_available, tmp_path):
        """Test that requesting GPU when CUDA unavailable causes exit"""
        log_file = open(tmp_path / "test.log", "w")
        with pytest.raises(SystemExit) as exc_info:
            parse_device("0", log_file)
        assert exc_info.value.code == 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_parse_device_invalid_gpu_index(
        self, mock_device_count, mock_cuda_available, tmp_path
    ):
        """Test that invalid GPU index is handled (should return device but log warning)"""
        log_file = open(tmp_path / "test.log", "w")
        # According to the code, it doesn't exit for invalid GPU index, just logs
        device = parse_device("5", log_file)
        # The function returns the device even if it's out of range
        assert device == 5
        log_file.close()


class TestPairedDataset:
    """Tests for PairedDataset class"""

    def test_paired_dataset_initialization(self):
        """Test basic initialization of PairedDataset"""
        X0 = [1, 2, 3]
        X1 = [4, 5, 6]
        Y = [0, 1, 0]

        dataset = PairedDataset(X0, X1, Y)

        assert dataset.X0 == X0
        assert dataset.X1 == X1
        assert dataset.Y == Y

    def test_paired_dataset_len(self):
        """Test __len__ method"""
        X0 = [1, 2, 3, 4, 5]
        X1 = [6, 7, 8, 9, 10]
        Y = [0, 1, 0, 1, 0]

        dataset = PairedDataset(X0, X1, Y)
        assert len(dataset) == 5

    def test_paired_dataset_getitem(self):
        """Test __getitem__ method"""
        X0 = ["seq1", "seq2", "seq3"]
        X1 = ["seq4", "seq5", "seq6"]
        Y = [torch.tensor(0), torch.tensor(1), torch.tensor(0)]

        dataset = PairedDataset(X0, X1, Y)

        # Test individual items
        item0 = dataset[0]
        assert item0 == ("seq1", "seq4", torch.tensor(0))

        item1 = dataset[1]
        assert item1 == ("seq2", "seq5", torch.tensor(1))

    def test_paired_dataset_mismatched_x0_x1(self):
        """Test that mismatched X0 and X1 lengths raise assertion error"""
        X0 = [1, 2, 3]
        X1 = [4, 5]  # Different length
        Y = [0, 1, 0]

        with pytest.raises(AssertionError):
            PairedDataset(X0, X1, Y)

    def test_paired_dataset_mismatched_x_y(self):
        """Test that mismatched X and Y lengths raise assertion error"""
        X0 = [1, 2, 3]
        X1 = [4, 5, 6]
        Y = [0, 1]  # Different length

        with pytest.raises(AssertionError):
            PairedDataset(X0, X1, Y)

    def test_paired_dataset_empty(self):
        """Test PairedDataset with empty lists"""
        X0 = []
        X1 = []
        Y = []

        dataset = PairedDataset(X0, X1, Y)
        assert len(dataset) == 0

    def test_paired_dataset_with_tensors(self):
        """Test PairedDataset with tensor data"""
        X0 = [torch.randn(10) for _ in range(5)]
        X1 = [torch.randn(10) for _ in range(5)]
        Y = [torch.tensor(i % 2) for i in range(5)]

        dataset = PairedDataset(X0, X1, Y)
        assert len(dataset) == 5

        x0, x1, y = dataset[0]
        assert isinstance(x0, torch.Tensor)
        assert isinstance(x1, torch.Tensor)
        assert isinstance(y, torch.Tensor)


class TestCollatePairedSequences:
    """Tests for collate_paired_sequences function"""

    def test_collate_basic(self):
        """Test basic collation of paired sequences"""
        args = [
            ("seq1", "seq2", torch.tensor(0)),
            ("seq3", "seq4", torch.tensor(1)),
            ("seq5", "seq6", torch.tensor(0)),
        ]

        x0, x1, y = collate_paired_sequences(args)

        assert x0 == ["seq1", "seq3", "seq5"]
        assert x1 == ["seq2", "seq4", "seq6"]
        assert torch.equal(y, torch.tensor([0, 1, 0]))

    def test_collate_with_tensors(self):
        """Test collation with tensor sequences"""
        args = [
            (torch.randn(5), torch.randn(5), torch.tensor(0.0)),
            (torch.randn(5), torch.randn(5), torch.tensor(1.0)),
        ]

        x0, x1, y = collate_paired_sequences(args)

        assert len(x0) == 2
        assert len(x1) == 2
        assert y.shape == (2,)
        assert isinstance(y, torch.Tensor)

    def test_collate_single_item(self):
        """Test collation with a single item"""
        args = [("seq1", "seq2", torch.tensor(1))]

        x0, x1, y = collate_paired_sequences(args)

        assert x0 == ["seq1"]
        assert x1 == ["seq2"]
        assert torch.equal(y, torch.tensor([1]))

    def test_collate_empty(self):
        """Test collation with empty list"""
        args = []

        x0, x1, y = collate_paired_sequences(args)

        assert x0 == []
        assert x1 == []
        assert y.shape == (0,)

    def test_collate_preserves_order(self):
        """Test that collation preserves order"""
        args = [
            (f"seq{i}", f"seq{i+10}", torch.tensor(i))
            for i in range(10)
        ]

        x0, x1, y = collate_paired_sequences(args)

        # Check order is preserved
        for i in range(10):
            assert x0[i] == f"seq{i}"
            assert x1[i] == f"seq{i+10}"
            assert y[i] == i
