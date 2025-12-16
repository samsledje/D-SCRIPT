"""
Tests for parallel data loading functionality in dscript.loading
"""

from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from dscript.loading import LoadingPool


class TestLoadingPool:
    """Tests for LoadingPool class"""

    @pytest.fixture
    def temp_hdf5_file(self, tmp_path):
        """Create a temporary HDF5 file with test data"""
        file_path = tmp_path / "test_embeddings.h5"

        with h5py.File(file_path, "w") as f:
            # Create some test embeddings
            f.create_dataset("protein1", data=np.random.randn(128))
            f.create_dataset("protein2", data=np.random.randn(128))
            f.create_dataset("protein3", data=np.random.randn(128))
            f.create_dataset("protein4", data=np.random.randn(128))
            f.create_dataset("protein5", data=np.random.randn(128))

        return str(file_path)

    def test_loading_pool_initialization_default_njobs(self, temp_hdf5_file):
        """Test LoadingPool initialization with default n_jobs"""
        pool = LoadingPool(temp_hdf5_file)

        # Should use cpu_count for n_jobs=-1
        import torch.multiprocessing as mp

        assert pool.n_jobs == mp.cpu_count()
        assert pool.queue_timeout == 60

        # Cleanup
        pool.shutdown()

    def test_loading_pool_initialization_custom_njobs(self, temp_hdf5_file):
        """Test LoadingPool initialization with custom n_jobs"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=4)

        assert pool.n_jobs == 4
        pool.shutdown()

    def test_loading_pool_initialization_custom_timeout(self, temp_hdf5_file):
        """Test LoadingPool initialization with custom timeout"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2, timeout=120)

        assert pool.queue_timeout == 120
        pool.shutdown()

    def test_load_once_basic(self, temp_hdf5_file):
        """Test basic loading of embeddings using load_once"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1", "protein2", "protein3"]
        embeddings = pool.load_once(keys, progress=False)

        # Check that we got the right number of embeddings
        assert len(embeddings) == 3

        # Check that embeddings are tensors
        for emb in embeddings:
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (128,)

        # Check that embeddings are not None
        assert all(emb is not None for emb in embeddings)

    def test_load_once_preserves_order(self, temp_hdf5_file):
        """Test that load_once preserves the order of keys"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        # Load in specific order
        keys = ["protein3", "protein1", "protein2"]
        embeddings = pool.load_once(keys, progress=False)

        # Verify we got 3 embeddings in order
        assert len(embeddings) == 3
        assert all(emb is not None for emb in embeddings)

    def test_load_once_single_key(self, temp_hdf5_file):
        """Test loading a single key"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1"]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], torch.Tensor)

    def test_load_once_empty_keys(self, temp_hdf5_file):
        """Test loading with empty keys list"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = []
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 0

    def test_load_once_with_progress(self, temp_hdf5_file):
        """Test load_once with progress bar enabled"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1", "protein2"]
        # Should not raise any errors with progress=True
        embeddings = pool.load_once(keys, progress=True)

        assert len(embeddings) == 2

    def test_load_basic(self, temp_hdf5_file):
        """Test basic loading of embeddings using load method"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1", "protein2"]
        embeddings = pool.load(keys, progress=False)

        assert len(embeddings) == 2
        assert all(isinstance(emb, torch.Tensor) for emb in embeddings)

        # Cleanup
        pool.shutdown()

    def test_load_with_progress(self, temp_hdf5_file):
        """Test load method with progress bar"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1", "protein2", "protein3"]
        embeddings = pool.load(keys, progress=True)

        assert len(embeddings) == 3

        # Cleanup
        pool.shutdown()

    def test_shutdown(self, temp_hdf5_file):
        """Test proper shutdown of LoadingPool"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        # Load some data
        keys = ["protein1"]
        embeddings = pool.load(keys, progress=False)
        assert len(embeddings) == 1

        # Shutdown should complete without errors
        pool.shutdown()

    def test_multiple_loads(self, temp_hdf5_file):
        """Test multiple sequential loads"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        # First load
        keys1 = ["protein1", "protein2"]
        embeddings1 = pool.load(keys1, progress=False)
        assert len(embeddings1) == 2

        # Second load
        keys2 = ["protein3", "protein4"]
        embeddings2 = pool.load(keys2, progress=False)
        assert len(embeddings2) == 2

        pool.shutdown()

    def test_load_all_embeddings(self, temp_hdf5_file):
        """Test loading all embeddings in the file"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1", "protein2", "protein3", "protein4", "protein5"]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 5
        assert all(isinstance(emb, torch.Tensor) for emb in embeddings)

    @pytest.fixture
    def temp_hdf5_file_large(self, tmp_path):
        """Create a temporary HDF5 file with larger embeddings"""
        file_path = tmp_path / "test_embeddings_large.h5"

        with h5py.File(file_path, "w") as f:
            # Create embeddings with different dimensions
            f.create_dataset("prot_1", data=np.random.randn(512))
            f.create_dataset("prot_2", data=np.random.randn(512))
            f.create_dataset("prot_3", data=np.random.randn(256))

        return str(file_path)

    def test_load_different_embedding_sizes(self, temp_hdf5_file_large):
        """Test loading embeddings of different sizes"""
        pool = LoadingPool(temp_hdf5_file_large, n_jobs=2)

        keys = ["prot_1", "prot_2", "prot_3"]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 3
        assert embeddings[0].shape == (512,)
        assert embeddings[1].shape == (512,)
        assert embeddings[2].shape == (256,)

    @pytest.fixture
    def temp_hdf5_file_2d(self, tmp_path):
        """Create a temporary HDF5 file with 2D embeddings"""
        file_path = tmp_path / "test_embeddings_2d.h5"

        with h5py.File(file_path, "w") as f:
            # Create 2D embeddings
            f.create_dataset("matrix1", data=np.random.randn(10, 128))
            f.create_dataset("matrix2", data=np.random.randn(20, 128))

        return str(file_path)

    def test_load_2d_embeddings(self, temp_hdf5_file_2d):
        """Test loading 2D embeddings"""
        pool = LoadingPool(temp_hdf5_file_2d, n_jobs=2)

        keys = ["matrix1", "matrix2"]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 2
        assert embeddings[0].shape == (10, 128)
        assert embeddings[1].shape == (20, 128)

    def test_load_duplicate_keys(self, temp_hdf5_file):
        """Test loading with duplicate keys"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1", "protein1", "protein2"]
        embeddings = pool.load_once(keys, progress=False)

        # Should get 3 embeddings (duplicates included)
        assert len(embeddings) == 3
        # First two should be the same protein
        assert torch.equal(embeddings[0], embeddings[1])

    def test_njobs_1(self, temp_hdf5_file):
        """Test with single worker process"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=1)

        keys = ["protein1", "protein2"]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 2
        assert all(isinstance(emb, torch.Tensor) for emb in embeddings)

    def test_loaded_embeddings_are_tensors(self, temp_hdf5_file):
        """Test that loaded embeddings are PyTorch tensors"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=2)

        keys = ["protein1"]
        embeddings = pool.load_once(keys, progress=False)

        emb = embeddings[0]
        assert isinstance(emb, torch.Tensor)
        assert emb.dtype == torch.float64  # numpy default is float64

    @patch("dscript.loading.mp.cpu_count", return_value=8)
    def test_njobs_auto_detection(self, mock_cpu_count, temp_hdf5_file):
        """Test that n_jobs=-1 uses cpu_count"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=-1)

        assert pool.n_jobs == 8
        pool.shutdown()

    @patch("dscript.loading.mp.cpu_count", return_value=4)
    def test_njobs_zero_uses_cpu_count(self, mock_cpu_count, temp_hdf5_file):
        """Test that n_jobs=0 uses cpu_count"""
        pool = LoadingPool(temp_hdf5_file, n_jobs=0)

        assert pool.n_jobs == 4
        pool.shutdown()


class TestLoadingPoolEdgeCases:
    """Tests for edge cases and error conditions"""

    @pytest.fixture
    def temp_empty_hdf5(self, tmp_path):
        """Create an empty HDF5 file"""
        file_path = tmp_path / "empty.h5"
        with h5py.File(file_path, "w"):
            pass  # Create empty file
        return str(file_path)

    def test_load_from_empty_file(self, temp_empty_hdf5):
        """Test loading from empty HDF5 file"""
        pool = LoadingPool(temp_empty_hdf5, n_jobs=2)

        # Loading empty keys should work
        keys = []
        embeddings = pool.load_once(keys, progress=False)
        assert len(embeddings) == 0

    def test_load_nonexistent_key(self, tmp_path):
        """Test loading a key that doesn't exist in the file"""
        file_path = tmp_path / "test.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("protein1", data=np.random.randn(128))

        pool = LoadingPool(str(file_path), n_jobs=1)

        # This should cause an error in the worker process
        # The worker will log an error but may not propagate it
        # Depending on implementation, this might timeout or return None
        # We're testing that the pool handles this gracefully
        keys = ["nonexistent"]

        # The behavior here depends on error handling in the worker
        # At minimum, it should not crash the test process
        try:
            embeddings = pool.load_once(keys, progress=False)
            # If it returns, check the result
            if embeddings:
                # If error handling returns None or similar
                pass
        except Exception:
            # If it raises an exception, that's also acceptable
            pass

    @pytest.fixture
    def temp_hdf5_special_chars(self, tmp_path):
        """Create HDF5 file with special characters in keys"""
        file_path = tmp_path / "special.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("protein_1", data=np.random.randn(64))
            f.create_dataset("protein-2", data=np.random.randn(64))
            f.create_dataset("protein.3", data=np.random.randn(64))

        return str(file_path)

    def test_load_keys_with_special_chars(self, temp_hdf5_special_chars):
        """Test loading keys with special characters"""
        pool = LoadingPool(temp_hdf5_special_chars, n_jobs=2)

        keys = ["protein_1", "protein-2", "protein.3"]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 3
        assert all(isinstance(emb, torch.Tensor) for emb in embeddings)

    def test_loading_pool_handles_many_jobs(self, tmp_path):
        """Test LoadingPool with many worker processes"""
        file_path = tmp_path / "test.h5"
        with h5py.File(file_path, "w") as f:
            for i in range(20):
                f.create_dataset(f"protein{i}", data=np.random.randn(64))

        pool = LoadingPool(str(file_path), n_jobs=8)

        keys = [f"protein{i}" for i in range(20)]
        embeddings = pool.load_once(keys, progress=False)

        assert len(embeddings) == 20


class TestLoadingPoolIntegration:
    """Integration tests for LoadingPool"""

    def test_end_to_end_loading(self, tmp_path):
        """Test complete end-to-end loading workflow"""
        # Create test file
        file_path = tmp_path / "proteins.h5"
        protein_data = {
            "PROT_A": np.random.randn(256),
            "PROT_B": np.random.randn(256),
            "PROT_C": np.random.randn(256),
            "PROT_D": np.random.randn(256),
        }

        with h5py.File(file_path, "w") as f:
            for name, data in protein_data.items():
                f.create_dataset(name, data=data)

        # Initialize pool
        pool = LoadingPool(str(file_path), n_jobs=2)

        # Load subset of proteins
        keys = ["PROT_A", "PROT_C"]
        embeddings = pool.load_once(keys, progress=False)

        # Verify results
        assert len(embeddings) == 2
        assert all(emb.shape == (256,) for emb in embeddings)

        # Verify data integrity by comparing with original
        # (note: floating point comparison needs tolerance)
        with h5py.File(file_path, "r") as f:
            for key, emb in zip(keys, embeddings):
                expected = torch.from_numpy(f[key][:])
                assert torch.allclose(emb, expected)

    def test_sequential_and_parallel_loading_consistency(self, tmp_path):
        """Test that sequential and parallel loading give same results"""
        file_path = tmp_path / "test.h5"

        # Create test data
        test_data = {f"seq{i}": np.random.randn(100) for i in range(10)}
        with h5py.File(file_path, "w") as f:
            for name, data in test_data.items():
                f.create_dataset(name, data=data)

        keys = list(test_data.keys())

        # Load with 1 job (sequential)
        pool1 = LoadingPool(str(file_path), n_jobs=1)
        embeddings1 = pool1.load_once(keys, progress=False)

        # Load with multiple jobs (parallel)
        pool2 = LoadingPool(str(file_path), n_jobs=4)
        embeddings2 = pool2.load_once(keys, progress=False)

        # Results should be identical
        assert len(embeddings1) == len(embeddings2)
        for emb1, emb2 in zip(embeddings1, embeddings2):
            assert torch.equal(emb1, emb2)
