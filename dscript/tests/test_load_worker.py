"""
Tests for HDF5 loading worker in dscript.load_worker
"""

import queue
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from dscript.load_worker import _hdf5_load_partial_func


class TestHDF5LoadWorker:
    """Tests for _hdf5_load_partial_func worker function"""

    @pytest.fixture
    def temp_hdf5_file(self, tmp_path):
        """Create a temporary HDF5 file with test embeddings"""
        file_path = tmp_path / "test_embeddings.h5"

        with h5py.File(file_path, "w") as f:
            # Create various embeddings
            f.create_dataset("protein1", data=np.random.randn(100, 128))
            f.create_dataset("protein2", data=np.random.randn(150, 128))
            f.create_dataset("protein3", data=np.random.randn(200, 256))
            f.create_dataset("special_protein", data=np.random.randn(50, 64))

        return str(file_path)

    def test_worker_basic_functionality(self, temp_hdf5_file):
        """Test basic worker functionality"""
        qin = queue.Queue()
        qout = queue.Queue()

        # Add work items: (key, index)
        qin.put(("protein1", 0))
        qin.put(("protein2", 1))
        qin.put(None)  # Sentinel to stop

        # Run worker
        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Collect results
        results = []
        while not qout.empty():
            result = qout.get()
            if result is not None:
                results.append(result)

        # Should have 2 results + 1 None sentinel
        assert len(results) == 2

        # Check results structure: (index, tensor)
        indices = [r[0] for r in results]
        tensors = [r[1] for r in results]

        assert 0 in indices
        assert 1 in indices

        # Check that tensors are correct type
        for tensor in tensors:
            assert isinstance(tensor, torch.Tensor)

    def test_worker_loads_correct_shapes(self, temp_hdf5_file):
        """Test that worker loads embeddings with correct shapes"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("protein1", 0))
        qin.put(("protein3", 1))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Collect results
        results = {}
        while not qout.empty():
            result = qout.get()
            if result is not None:
                idx, tensor = result
                results[idx] = tensor

        # Check shapes
        assert results[0].shape == (100, 128)  # protein1
        assert results[1].shape == (200, 256)  # protein3

    def test_worker_preserves_order_indices(self, temp_hdf5_file):
        """Test that worker preserves index ordering"""
        qin = queue.Queue()
        qout = queue.Queue()

        # Add in specific order with specific indices
        qin.put(("protein1", 5))
        qin.put(("protein2", 3))
        qin.put(("protein3", 1))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Collect all non-None results
        results = []
        while not qout.empty():
            result = qout.get()
            if result is not None:
                results.append(result)

        # Check that indices are preserved
        indices = [r[0] for r in results]
        assert 5 in indices
        assert 3 in indices
        assert 1 in indices

    def test_worker_sends_sentinel(self, temp_hdf5_file):
        """Test that worker sends None sentinel when done"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("protein1", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Last item should be None
        results = []
        while not qout.empty():
            results.append(qout.get())

        assert results[-1] is None

    def test_worker_handles_multiple_loads(self, temp_hdf5_file):
        """Test worker handling multiple sequential loads"""
        qin = queue.Queue()
        qout = queue.Queue()

        # Load multiple embeddings
        for i, protein in enumerate(
            ["protein1", "protein2", "protein3", "special_protein"]
        ):
            qin.put((protein, i))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Should have 4 results + 1 None
        results = []
        while not qout.empty():
            result = qout.get()
            if result is not None:
                results.append(result)

        assert len(results) == 4

    def test_worker_converts_numpy_to_torch(self, temp_hdf5_file):
        """Test that worker converts numpy arrays to torch tensors"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("protein1", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        result = qout.get()
        assert result is not None

        _, tensor = result
        assert isinstance(tensor, torch.Tensor)
        assert not isinstance(tensor, np.ndarray)

    def test_worker_shares_memory(self, temp_hdf5_file):
        """Test that loaded tensors have shared memory enabled"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("protein1", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        result = qout.get()
        _, tensor = result

        # Tensor should be in shared memory
        assert tensor.is_shared()

    @patch("torch.set_num_threads")
    def test_worker_sets_num_threads(self, mock_set_threads, temp_hdf5_file):
        """Test that worker sets torch threads to 1"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("protein1", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Should set threads to 1
        mock_set_threads.assert_called_once_with(1)

    def test_worker_handles_empty_queue(self, temp_hdf5_file):
        """Test worker with only sentinel (no actual work)"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(None)  # Just the sentinel

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Should only have the None sentinel
        result = qout.get()
        assert result is None
        assert qout.empty()

    def test_worker_preserves_data_values(self, tmp_path):
        """Test that worker preserves actual data values"""
        file_path = tmp_path / "test.h5"

        # Create file with known data
        test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with h5py.File(file_path, "w") as f:
            f.create_dataset("test", data=test_data)

        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("test", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, str(file_path))

        _, tensor = qout.get()

        # Check values are preserved
        assert torch.allclose(tensor, torch.from_numpy(test_data))

    @patch("dscript.load_worker.logger")
    def test_worker_logs_errors(self, mock_logger, tmp_path):
        """Test that worker logs errors for missing keys"""
        file_path = tmp_path / "test.h5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset("exists", data=np.random.randn(10))

        qin = queue.Queue()
        qout = queue.Queue()

        # Request a key that doesn't exist
        qin.put(("nonexistent", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, str(file_path))

        # Should have logged an error
        assert mock_logger.error.called

    def test_worker_with_different_dtypes(self, tmp_path):
        """Test worker with different numpy dtypes"""
        file_path = tmp_path / "test.h5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset("float32", data=np.random.randn(10).astype(np.float32))
            f.create_dataset("float64", data=np.random.randn(10).astype(np.float64))
            f.create_dataset("int32", data=np.arange(10, dtype=np.int32))

        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("float32", 0))
        qin.put(("float64", 1))
        qin.put(("int32", 2))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, str(file_path))

        # Collect results
        results = {}
        while not qout.empty():
            result = qout.get()
            if result is not None:
                idx, tensor = result
                results[idx] = tensor

        # All should be converted to tensors
        assert all(isinstance(t, torch.Tensor) for t in results.values())

    def test_worker_handles_1d_arrays(self, tmp_path):
        """Test worker with 1D arrays"""
        file_path = tmp_path / "test.h5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset("1d_array", data=np.random.randn(128))

        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("1d_array", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, str(file_path))

        _, tensor = qout.get()

        assert tensor.ndim == 1
        assert tensor.shape == (128,)

    def test_worker_handles_3d_arrays(self, tmp_path):
        """Test worker with 3D arrays"""
        file_path = tmp_path / "test.h5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset("3d_array", data=np.random.randn(10, 20, 30))

        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("3d_array", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, str(file_path))

        _, tensor = qout.get()

        assert tensor.ndim == 3
        assert tensor.shape == (10, 20, 30)

    def test_worker_sequential_processing(self, temp_hdf5_file):
        """Test that worker processes items sequentially from queue"""
        qin = queue.Queue()
        qout = queue.Queue()

        # Add items in specific order
        order = [
            ("protein1", 0),
            ("protein2", 1),
            ("protein3", 2),
            ("special_protein", 3),
        ]
        for item in order:
            qin.put(item)
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, temp_hdf5_file)

        # Results should be in same order (excluding None sentinel)
        results = []
        while not qout.empty():
            result = qout.get()
            if result is not None:
                results.append(result)

        # Should have processed all items
        assert len(results) == 4

        # Indices should match input order
        indices = [r[0] for r in results]
        assert indices == [0, 1, 2, 3]


class TestLoadWorkerEdgeCases:
    """Edge case tests for load worker"""

    def test_worker_with_corrupted_file(self, tmp_path):
        """Test worker behavior with corrupted HDF5 file"""
        file_path = tmp_path / "corrupted.h5"
        file_path.write_text("NOT A VALID HDF5 FILE")

        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("anything", 0))
        qin.put(None)

        # Should handle error gracefully
        with patch("dscript.load_worker.logger") as mock_logger:
            _hdf5_load_partial_func(qin, qout, str(file_path))
            assert mock_logger.error.called

    def test_worker_with_nonexistent_file(self):
        """Test worker with non-existent file path"""
        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("key", 0))
        qin.put(None)

        # Should handle error gracefully
        with patch("dscript.load_worker.logger") as mock_logger:
            _hdf5_load_partial_func(qin, qout, "/nonexistent/path.h5")
            assert mock_logger.error.called

    def test_worker_large_embedding(self, tmp_path):
        """Test worker with very large embedding"""
        file_path = tmp_path / "large.h5"

        # Create a large embedding
        large_data = np.random.randn(10000, 1024)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("large", data=large_data)

        qin = queue.Queue()
        qout = queue.Queue()

        qin.put(("large", 0))
        qin.put(None)

        _hdf5_load_partial_func(qin, qout, str(file_path))

        _, tensor = qout.get()

        assert tensor.shape == (10000, 1024)
        assert tensor.is_shared()  # Should still be in shared memory
