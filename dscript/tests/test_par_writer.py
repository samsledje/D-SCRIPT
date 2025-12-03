"""
Tests for parallel prediction writer in dscript.commands.par_writer
"""

import queue
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

from dscript.commands.par_writer import _writer


class TestParWriter:
    """Tests for _writer function"""

    @pytest.fixture
    def basic_setup(self, tmp_path):
        """Create basic test setup with temp files"""
        all_prots = ["prot1", "prot2", "prot3", "prot4"]
        out_all = tmp_path / "all_predictions.tsv"
        out_pos = tmp_path / "positive_predictions.tsv"
        threshold = 0.5

        return {
            "all_prots": all_prots,
            "out_all": str(out_all),
            "out_pos": str(out_pos),
            "threshold": threshold,
        }

    def test_writer_basic_functionality(self, basic_setup, tmp_path):
        """Test basic writer functionality without contact maps"""
        # Create a queue with predictions
        output_queue = queue.Queue()

        # Add some predictions (i0, i1, probability)
        predictions = [
            (0, 1, 0.8),  # Above threshold
            (0, 2, 0.3),  # Below threshold
            (1, 2, 0.9),  # Above threshold
            (2, 3, 0.1),  # Below threshold
        ]

        for pred in predictions:
            output_queue.put(pred)

        # Run writer
        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,  # No contact maps
            len(predictions),
            basic_setup["threshold"],
            output_queue,
        )

        # Verify all predictions file
        with open(basic_setup["out_all"]) as f:
            all_lines = f.readlines()

        assert len(all_lines) == 4
        assert "prot1\tprot2\t0.8\n" in all_lines
        assert "prot1\tprot3\t0.3\n" in all_lines
        assert "prot2\tprot3\t0.9\n" in all_lines
        assert "prot3\tprot4\t0.1\n" in all_lines

        # Verify positive predictions file (only >= threshold)
        with open(basic_setup["out_pos"]) as f:
            pos_lines = f.readlines()

        assert len(pos_lines) == 2
        assert "prot1\tprot2\t0.8\n" in pos_lines
        assert "prot2\tprot3\t0.9\n" in pos_lines

    def test_writer_threshold_filtering(self, basic_setup):
        """Test that threshold correctly filters predictions"""
        output_queue = queue.Queue()

        # Predictions with various scores
        predictions = [
            (0, 1, 0.49999),  # Just below threshold
            (0, 2, 0.5),  # Exactly at threshold
            (1, 2, 0.50001),  # Just above threshold
        ]

        for pred in predictions:
            output_queue.put(pred)

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            len(predictions),
            0.5,  # threshold
            output_queue,
        )

        # All predictions should be in all file
        with open(basic_setup["out_all"]) as f:
            all_lines = f.readlines()
        assert len(all_lines) == 3

        # Only >= 0.5 should be in positive file
        with open(basic_setup["out_pos"]) as f:
            pos_lines = f.readlines()

        assert len(pos_lines) == 2
        # Check that 0.49999 is not in positive
        assert "0.49999" not in "".join(pos_lines)
        # Check that 0.5 and 0.50001 are in positive
        assert any("0.5" in line for line in pos_lines)

    def test_writer_with_contact_maps(self, basic_setup, tmp_path):
        """Test writer with contact map storage"""
        cmap_path = tmp_path / "contact_maps.h5"
        output_queue = queue.Queue()

        # Create mock contact maps
        cm1 = torch.randn(50, 50)
        cm2 = torch.randn(60, 60)

        # Predictions with contact maps (i0, i1, p, contact_map)
        predictions = [
            (0, 1, 0.8, cm1),  # Above threshold - should store cmap
            (0, 2, 0.3, cm2),  # Below threshold - should not store cmap
        ]

        for pred in predictions:
            output_queue.put(pred)

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            str(cmap_path),  # Enable contact map storage
            len(predictions),
            basic_setup["threshold"],
            output_queue,
        )

        # Verify contact maps file was created
        assert cmap_path.exists()

        # Check contact maps
        with h5py.File(cmap_path, "r") as f:
            # Only positive prediction should have contact map
            assert "prot1xprot2" in f
            assert "prot1xprot3" not in f  # Below threshold

            # Verify contact map shape
            stored_cm = f["prot1xprot2"][:]
            assert stored_cm.shape == (50, 50)
            assert np.allclose(stored_cm, cm1.numpy())

    def test_writer_protein_name_mapping(self, tmp_path):
        """Test correct protein name mapping from indices"""
        all_prots = ["PROTEIN_A", "PROTEIN_B", "PROTEIN_C"]
        out_all = tmp_path / "all.tsv"
        out_pos = tmp_path / "pos.tsv"

        output_queue = queue.Queue()
        output_queue.put((0, 1, 0.9))
        output_queue.put((1, 2, 0.8))
        output_queue.put((0, 2, 0.7))

        _writer(all_prots, str(out_all), str(out_pos), None, 3, 0.5, output_queue)

        with open(out_all) as f:
            lines = f.readlines()

        # Check protein names are correctly mapped
        assert "PROTEIN_A\tPROTEIN_B\t0.9\n" in lines
        assert "PROTEIN_B\tPROTEIN_C\t0.8\n" in lines
        assert "PROTEIN_A\tPROTEIN_C\t0.7\n" in lines

    def test_writer_zero_predictions(self, basic_setup):
        """Test writer with zero predictions"""
        output_queue = queue.Queue()

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            0,  # Zero predictions
            basic_setup["threshold"],
            output_queue,
        )

        # Files should be created but empty
        with open(basic_setup["out_all"]) as f:
            assert f.read() == ""

        with open(basic_setup["out_pos"]) as f:
            assert f.read() == ""

    def test_writer_single_prediction(self, basic_setup):
        """Test writer with single prediction"""
        output_queue = queue.Queue()
        output_queue.put((0, 1, 0.95))

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            1,
            basic_setup["threshold"],
            output_queue,
        )

        with open(basic_setup["out_all"]) as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert "prot1\tprot2\t0.95\n" in lines

    def test_writer_all_below_threshold(self, basic_setup):
        """Test when all predictions are below threshold"""
        output_queue = queue.Queue()

        predictions = [
            (0, 1, 0.1),
            (0, 2, 0.2),
            (1, 2, 0.3),
        ]

        for pred in predictions:
            output_queue.put(pred)

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            len(predictions),
            basic_setup["threshold"],
            output_queue,
        )

        # All file should have all predictions
        with open(basic_setup["out_all"]) as f:
            assert len(f.readlines()) == 3

        # Positive file should be empty
        with open(basic_setup["out_pos"]) as f:
            assert f.read() == ""

    def test_writer_all_above_threshold(self, basic_setup):
        """Test when all predictions are above threshold"""
        output_queue = queue.Queue()

        predictions = [
            (0, 1, 0.9),
            (0, 2, 0.8),
            (1, 2, 0.7),
        ]

        for pred in predictions:
            output_queue.put(pred)

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            len(predictions),
            basic_setup["threshold"],
            output_queue,
        )

        # Both files should have all predictions
        with open(basic_setup["out_all"]) as f:
            all_lines = f.readlines()

        with open(basic_setup["out_pos"]) as f:
            pos_lines = f.readlines()

        assert len(all_lines) == len(pos_lines) == 3

    def test_writer_contact_map_names(self, basic_setup, tmp_path):
        """Test contact map dataset naming convention"""
        cmap_path = tmp_path / "cmaps.h5"
        output_queue = queue.Queue()

        cm = torch.randn(10, 10)
        output_queue.put((0, 1, 0.9, cm))
        output_queue.put((2, 3, 0.8, cm))

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            str(cmap_path),
            2,
            basic_setup["threshold"],
            output_queue,
        )

        with h5py.File(cmap_path, "r") as f:
            # Check naming convention: protein1xprotein2
            assert "prot1xprot2" in f
            assert "prot3xprot4" in f

    def test_writer_large_contact_maps(self, basic_setup, tmp_path):
        """Test handling of large contact maps"""
        cmap_path = tmp_path / "large_cmaps.h5"
        output_queue = queue.Queue()

        # Large contact map
        large_cm = torch.randn(500, 500)
        output_queue.put((0, 1, 0.9, large_cm))

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            str(cmap_path),
            1,
            basic_setup["threshold"],
            output_queue,
        )

        with h5py.File(cmap_path, "r") as f:
            stored = f["prot1xprot2"][:]
            assert stored.shape == (500, 500)

    def test_writer_different_sized_contact_maps(self, basic_setup, tmp_path):
        """Test handling contact maps of different sizes"""
        cmap_path = tmp_path / "cmaps.h5"
        output_queue = queue.Queue()

        cm1 = torch.randn(50, 60)
        cm2 = torch.randn(100, 100)
        cm3 = torch.randn(30, 40)

        output_queue.put((0, 1, 0.9, cm1))
        output_queue.put((0, 2, 0.8, cm2))
        output_queue.put((1, 2, 0.7, cm3))

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            str(cmap_path),
            3,
            basic_setup["threshold"],
            output_queue,
        )

        with h5py.File(cmap_path, "r") as f:
            assert f["prot1xprot2"][:].shape == (50, 60)
            assert f["prot1xprot3"][:].shape == (100, 100)
            assert f["prot2xprot3"][:].shape == (30, 40)

    def test_writer_prediction_order_preserved(self, basic_setup):
        """Test that prediction order is preserved in output"""
        output_queue = queue.Queue()

        # Add predictions in specific order
        predictions = [(i, i + 1, 0.5 + i * 0.1) for i in range(3)]

        for pred in predictions:
            output_queue.put(pred)

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            len(predictions),
            basic_setup["threshold"],
            output_queue,
        )

        with open(basic_setup["out_all"]) as f:
            lines = f.readlines()

        # Verify order is preserved
        assert lines[0].startswith("prot1\tprot2")
        assert lines[1].startswith("prot2\tprot3")
        assert lines[2].startswith("prot3\tprot4")

    def test_writer_float_precision(self, basic_setup):
        """Test that float precision is maintained in output"""
        output_queue = queue.Queue()

        # High precision float
        output_queue.put((0, 1, 0.123456789))

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            1,
            basic_setup["threshold"],
            output_queue,
        )

        with open(basic_setup["out_all"]) as f:
            content = f.read()

        # Check that precision is preserved
        assert "0.123456789" in content

    @patch("dscript.commands.par_writer.tqdm")
    def test_writer_progress_bar(self, mock_tqdm, basic_setup):
        """Test that progress bar is used correctly"""
        output_queue = queue.Queue()

        predictions = [(0, 1, 0.9), (0, 2, 0.8), (1, 2, 0.7)]
        for pred in predictions:
            output_queue.put(pred)

        # Setup mock
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            None,
            len(predictions),
            basic_setup["threshold"],
            output_queue,
        )

        # Verify tqdm was called with correct total
        mock_tqdm.assert_called_once()
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs["total"] == 3
        assert call_kwargs["desc"] == "Writing Predictions"

        # Verify update was called for each prediction
        assert mock_pbar.update.call_count == 3

    def test_writer_contact_map_dtype(self, basic_setup, tmp_path):
        """Test that contact maps are stored as float32"""
        cmap_path = tmp_path / "cmaps.h5"
        output_queue = queue.Queue()

        cm = torch.randn(10, 10).double()  # Use double precision
        output_queue.put((0, 1, 0.9, cm))

        _writer(
            basic_setup["all_prots"],
            basic_setup["out_all"],
            basic_setup["out_pos"],
            str(cmap_path),
            1,
            basic_setup["threshold"],
            output_queue,
        )

        with h5py.File(cmap_path, "r") as f:
            # Should be stored as float32
            assert f["prot1xprot2"].dtype == np.float32

    def test_writer_handles_protein_names_with_special_chars(self, tmp_path):
        """Test handling protein names with special characters"""
        all_prots = ["prot_1", "prot-2", "prot.3"]
        out_all = tmp_path / "all.tsv"
        out_pos = tmp_path / "pos.tsv"

        output_queue = queue.Queue()
        output_queue.put((0, 1, 0.9))

        _writer(all_prots, str(out_all), str(out_pos), None, 1, 0.5, output_queue)

        with open(out_all) as f:
            content = f.read()

        # Should handle special characters correctly
        assert "prot_1\tprot-2\t0.9\n" in content
