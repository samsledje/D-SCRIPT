import os
import tempfile
from io import StringIO

import pytest
from Bio import SeqIO

from dscript.fasta import (
    parse,
    parse_directory,
    write,
)


class TestFasta:
    """Test cases for fasta.py module."""

    def test_parse_from_string(self):
        """Test parsing FASTA sequences from a string."""
        fasta_content = """>seq1
ACGT
>seq2
TGCA"""

        fasta_io = StringIO(fasta_content)
        names, sequences = parse(fasta_io)

        assert len(names) == 2
        assert names == ["seq1", "seq2"]
        assert sequences == ["ACGT", "TGCA"]

    def test_parse_existing_file(self):
        """Test parsing FASTA sequences from existing test file."""
        names, sequences = parse("dscript/tests/test.fasta")

        assert len(names) == 3
        assert "seq1" in names
        assert "seq2" in names
        assert "seq3" in names
        assert len(sequences) == 3
        # Check that all sequences are strings
        for seq in sequences:
            assert isinstance(seq, str)
            assert len(seq) > 0

    def test_parse_with_comment(self):
        """Test parsing with comment parameter (though not used in current implementation)."""
        fasta_content = """>seq1
ACGT
>seq2
TGCA"""

        fasta_io = StringIO(fasta_content)
        names, sequences = parse(fasta_io, comment="#")

        assert len(names) == 2
        assert names == ["seq1", "seq2"]
        assert sequences == ["ACGT", "TGCA"]

    def test_parse_empty_file(self):
        """Test parsing an empty FASTA file."""
        fasta_content = ""
        fasta_io = StringIO(fasta_content)
        names, sequences = parse(fasta_io)

        assert len(names) == 0
        assert len(sequences) == 0

    def test_parse_directory(self):
        """Test parsing FASTA files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test sequence files
            seq_file1 = os.path.join(tmpdir, "test1.seq")
            seq_file2 = os.path.join(tmpdir, "test2.seq")

            with open(seq_file1, "wb") as f:
                f.write(b">seq1\nACGT")

            with open(seq_file2, "wb") as f:
                f.write(b">seq2\nTGCA")

            names, sequences = parse_directory(tmpdir, extension=".seq")

            assert len(names) == 2
            assert "seq1" in names
            assert "seq2" in names
            assert "ACGT" in sequences
            assert "TGCA" in sequences

    def test_parse_directory_no_matching_files(self):
        """Test parsing directory with no matching extension files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with different extension
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("not a sequence file")

            names, sequences = parse_directory(tmpdir, extension=".seq")

            assert len(names) == 0
            assert len(sequences) == 0

    def test_write_sequences(self):
        """Test writing sequences to FASTA file."""
        names = ["seq1", "seq2", "seq3"]
        sequences = ["ACGT", "TGCA", "AAAA"]

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".fasta", delete=False
        ) as tmp_file:
            write(names, sequences, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # Read back the written file
            parsed_names, parsed_sequences = parse(tmp_file_path)

            assert len(parsed_names) == 3
            assert parsed_names == names
            assert parsed_sequences == sequences
        finally:
            os.unlink(tmp_file_path)

    def test_write_empty_sequences(self):
        """Test writing empty sequences."""
        names = []
        sequences = []

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".fasta", delete=False
        ) as tmp_file:
            write(names, sequences, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # Read back the written file
            parsed_names, parsed_sequences = parse(tmp_file_path)

            assert len(parsed_names) == 0
            assert len(parsed_sequences) == 0
        finally:
            os.unlink(tmp_file_path)

    def test_write_with_strict_zip(self):
        """Test write function handles strict zip correctly."""
        names = ["seq1", "seq2"]
        sequences = ["ACGT", "TGCA"]

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".fasta", delete=False
        ) as tmp_file:
            write(names, sequences, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # Verify the file was written correctly
            records = list(SeqIO.parse(tmp_file_path, "fasta"))
            assert len(records) == 2
            assert records[0].id == "seq1"
            assert str(records[0].seq) == "ACGT"
            assert records[1].id == "seq2"
            assert str(records[1].seq) == "TGCA"
        finally:
            os.unlink(tmp_file_path)

    def test_write_mismatched_lengths(self):
        """Test write function with mismatched name and sequence lengths."""
        names = ["seq1", "seq2", "seq3"]
        sequences = ["ACGT", "TGCA"]  # One less sequence than names

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".fasta") as tmp_file:
            # This should raise an error due to strict=False in zip
            with pytest.raises(ValueError):
                write(names, sequences, tmp_file)
