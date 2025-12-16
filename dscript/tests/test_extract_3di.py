"""
Tests for 3Di sequence extraction functionality in dscript.commands.extract_3di
"""

from pathlib import Path
from unittest.mock import Mock, patch

import biotite.sequence.io.fasta as fasta
import pytest

from dscript.commands.extract_3di import add_args, main


class TestExtract3Di:
    """Tests for extract_3di command"""

    @pytest.fixture
    def temp_pdb_dir(self, tmp_path):
        """Create a temporary directory with mock PDB files"""
        pdb_dir = tmp_path / "pdb_files"
        pdb_dir.mkdir()

        # Create some mock PDB files
        (pdb_dir / "protein1.pdb").write_text("MOCK PDB CONTENT 1")
        (pdb_dir / "protein2.pdb").write_text("MOCK PDB CONTENT 2")
        (pdb_dir / "protein3.cif").write_text("MOCK CIF CONTENT")
        (pdb_dir / "readme.txt").write_text("README")  # Non-PDB file

        return pdb_dir

    @pytest.fixture
    def mock_args(self, tmp_path, temp_pdb_dir):
        """Create mock arguments for main function"""
        args = Mock()
        args.pdb_directory = str(temp_pdb_dir)
        args.out_file = str(tmp_path / "output.fasta")
        return args

    def test_add_args_creates_parser(self):
        """Test that add_args configures parser correctly"""
        import argparse

        parser = argparse.ArgumentParser()
        result = add_args(parser)

        # Should return the parser
        assert result is parser

        # Should have added required arguments
        # We can't easily introspect argparse, but we can try parsing
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Should fail without required args

    def test_add_args_accepts_valid_arguments(self):
        """Test that parser accepts valid arguments"""
        import argparse

        parser = argparse.ArgumentParser()
        add_args(parser)

        args = parser.parse_args(["/path/to/pdb", "/path/to/output.fasta"])

        assert args.pdb_directory == "/path/to/pdb"
        assert args.out_file == "/path/to/output.fasta"

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_basic_execution(self, mock_get_3di, mock_args, temp_pdb_dir):
        """Test basic execution of main function"""
        # Setup mock to return some 3Di sequences
        mock_get_3di.return_value = {
            "protein1": "ABCDEFGHIJK",
            "protein2": "LMNOPQRSTUV",
            "protein3": "WXYZABCDEFG",
        }

        # Run main
        main(mock_args)

        # Verify get_3di_sequences was called with correct files
        call_args = mock_get_3di.call_args[0][0]
        pdb_files = [p.name for p in call_args]

        # Should include .pdb and .cif files but not .txt
        assert "protein1.pdb" in pdb_files
        assert "protein2.pdb" in pdb_files
        assert "protein3.cif" in pdb_files
        assert "readme.txt" not in pdb_files

        # Verify output file was created
        assert Path(mock_args.out_file).exists()

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_writes_correct_fasta(self, mock_get_3di, mock_args):
        """Test that main writes correct FASTA output"""
        # Setup mock sequences
        mock_sequences = {
            "PROT_A": "ABCDEFGHIJK",
            "PROT_B": "LMNOPQRSTUV",
        }
        mock_get_3di.return_value = mock_sequences

        # Run main
        main(mock_args)

        # Read and verify output FASTA
        with open(mock_args.out_file) as f:
            fasta_file = fasta.FastaFile.read(f)

        # Check that all sequences are in the output
        assert "PROT_A" in fasta_file
        assert "PROT_B" in fasta_file
        assert fasta_file["PROT_A"] == "ABCDEFGHIJK"
        assert fasta_file["PROT_B"] == "LMNOPQRSTUV"

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_with_empty_directory(self, mock_get_3di, tmp_path):
        """Test main with empty PDB directory"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        args = Mock()
        args.pdb_directory = str(empty_dir)
        args.out_file = str(tmp_path / "output.fasta")

        mock_get_3di.return_value = {}

        # Should not raise error with empty directory
        main(args)

        # get_3di_sequences should be called with empty list
        call_args = mock_get_3di.call_args[0][0]
        assert len(call_args) == 0

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_filters_file_extensions(self, mock_get_3di, tmp_path):
        """Test that only .pdb and .cif files are processed"""
        pdb_dir = tmp_path / "mixed_files"
        pdb_dir.mkdir()

        # Create various file types
        (pdb_dir / "valid1.pdb").write_text("PDB")
        (pdb_dir / "valid2.cif").write_text("CIF")
        (pdb_dir / "invalid.txt").write_text("TXT")
        (pdb_dir / "invalid.fasta").write_text("FASTA")
        (pdb_dir / "invalid.py").write_text("PYTHON")

        args = Mock()
        args.pdb_directory = str(pdb_dir)
        args.out_file = str(tmp_path / "output.fasta")

        mock_get_3di.return_value = {}

        main(args)

        # Only .pdb and .cif files should be passed
        call_args = mock_get_3di.call_args[0][0]
        file_names = [p.name for p in call_args]

        assert len(file_names) == 2
        assert "valid1.pdb" in file_names
        assert "valid2.cif" in file_names
        assert "invalid.txt" not in file_names
        assert "invalid.fasta" not in file_names
        assert "invalid.py" not in file_names

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_handles_pathlib_paths(self, mock_get_3di, tmp_path):
        """Test that main correctly handles Path objects"""
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        (pdb_dir / "test.pdb").write_text("PDB")

        args = Mock()
        args.pdb_directory = str(pdb_dir)
        args.out_file = str(tmp_path / "output.fasta")

        mock_get_3di.return_value = {"test": "ABCDEF"}

        main(args)

        # Verify Path objects were created correctly
        call_args = mock_get_3di.call_args[0][0]
        assert all(isinstance(p, Path) for p in call_args)

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_creates_output_file(self, mock_get_3di, tmp_path):
        """Test that output file is created if it doesn't exist"""
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        (pdb_dir / "test.pdb").write_text("PDB")

        output_file = tmp_path / "subdir" / "output.fasta"

        args = Mock()
        args.pdb_directory = str(pdb_dir)
        args.out_file = str(output_file)

        mock_get_3di.return_value = {"test": "ABCDEF"}

        # Parent directory doesn't exist yet
        assert not output_file.parent.exists()

        # Create parent directory (mimicking real usage)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        main(args)

        # Output file should be created
        assert output_file.exists()

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_with_multiple_proteins(self, mock_get_3di, tmp_path):
        """Test processing multiple proteins"""
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()

        # Create multiple PDB files
        for i in range(10):
            (pdb_dir / f"protein{i}.pdb").write_text(f"PDB {i}")

        args = Mock()
        args.pdb_directory = str(pdb_dir)
        args.out_file = str(tmp_path / "output.fasta")

        # Mock return with multiple sequences
        mock_sequences = {f"protein{i}": f"SEQ{i}" * 10 for i in range(10)}
        mock_get_3di.return_value = mock_sequences

        main(args)

        # Verify all sequences in output
        with open(args.out_file) as f:
            fasta_file = fasta.FastaFile.read(f)

        assert len(fasta_file) == 10
        for i in range(10):
            assert f"protein{i}" in fasta_file

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_preserves_sequence_content(self, mock_get_3di, tmp_path):
        """Test that sequence content is preserved correctly"""
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        (pdb_dir / "test.pdb").write_text("PDB")

        args = Mock()
        args.pdb_directory = str(pdb_dir)
        args.out_file = str(tmp_path / "output.fasta")

        # Use realistic 3Di alphabet characters
        test_sequence = "abcdefghijklmnopqrst"
        mock_get_3di.return_value = {"test_protein": test_sequence}

        main(args)

        # Read and verify
        with open(args.out_file) as f:
            fasta_file = fasta.FastaFile.read(f)

        assert fasta_file["test_protein"] == test_sequence

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_main_handles_special_characters_in_names(self, mock_get_3di, tmp_path):
        """Test handling of protein names with special characters"""
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        (pdb_dir / "test.pdb").write_text("PDB")

        args = Mock()
        args.pdb_directory = str(pdb_dir)
        args.out_file = str(tmp_path / "output.fasta")

        # Names with special characters
        mock_get_3di.return_value = {
            "protein_1": "ABCDEF",
            "protein-2": "GHIJKL",
            "protein.3": "MNOPQR",
        }

        main(args)

        # Should handle special characters in FASTA headers
        with open(args.out_file) as f:
            fasta_file = fasta.FastaFile.read(f)

        assert "protein_1" in fasta_file
        assert "protein-2" in fasta_file
        assert "protein.3" in fasta_file


class TestExtract3DiIntegration:
    """Integration tests for extract_3di"""

    @patch("dscript.commands.extract_3di.get_3di_sequences")
    def test_full_workflow(self, mock_get_3di, tmp_path):
        """Test full workflow from args to output"""
        import argparse

        # Setup
        pdb_dir = tmp_path / "structures"
        pdb_dir.mkdir()
        (pdb_dir / "1abc.pdb").write_text("PDB CONTENT")
        (pdb_dir / "2def.cif").write_text("CIF CONTENT")

        output_file = tmp_path / "3di_sequences.fasta"

        # Mock 3Di extraction
        mock_get_3di.return_value = {
            "1abc": "abcdefghijklmnopqrst",
            "2def": "uvwxyzabcdefghijklmn",
        }

        # Create parser and parse args
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([str(pdb_dir), str(output_file)])

        # Run main
        main(args)

        # Verify results
        assert output_file.exists()

        with open(output_file) as f:
            fasta_file = fasta.FastaFile.read(f)

        assert len(fasta_file) == 2
        assert fasta_file["1abc"] == "abcdefghijklmnopqrst"
        assert fasta_file["2def"] == "uvwxyzabcdefghijklmn"
