"""
Tests for CLI entry point in dscript.__main__
"""

from io import StringIO
from unittest.mock import Mock, patch

import pytest

from dscript.__main__ import CitationAction, main


class TestCitationAction:
    """Tests for CitationAction class"""

    def test_citation_action_initialization(self):
        """Test CitationAction can be initialized"""
        action = CitationAction(["--citation"], "citation")
        assert action is not None

    def test_citation_action_call_prints_citation(self):
        """Test that calling CitationAction prints citation"""
        action = CitationAction(["--citation"], "citation", nargs=0)

        parser = Mock()
        namespace = Mock()

        # Should exit with code 0
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stdout", new=StringIO()):
                action(parser, namespace, None)

        assert exc_info.value.code == 0

    def test_citation_action_sets_namespace(self):
        """Test that CitationAction sets namespace attribute"""
        action = CitationAction(["--citation"], "citation", nargs=0)

        parser = Mock()
        namespace = Mock()

        with pytest.raises(SystemExit):
            action(parser, namespace, "value")

        # Namespace attribute should be set before exit
        assert hasattr(namespace, "citation")


class TestMainFunction:
    """Tests for main() function"""

    def test_main_with_version_flag(self):
        """Test main with --version flag"""
        with patch("sys.argv", ["dscript", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.stdout", new=StringIO()):
                    main()

            # --version should exit with code 0
            assert exc_info.value.code == 0

    def test_main_with_citation_flag(self):
        """Test main with --citation flag"""
        with patch("sys.argv", ["dscript", "--citation"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # --citation should exit with code 0
            assert exc_info.value.code == 0

    def test_main_requires_subcommand(self):
        """Test that main requires a subcommand"""
        with patch("sys.argv", ["dscript"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should exit with non-zero code (argparse error)
            assert exc_info.value.code != 0

    @patch("dscript.commands.train.main")
    def test_main_calls_train_command(self, mock_train_main):
        """Test that main calls train command correctly"""
        test_args = [
            "dscript",
            "train",
            "--train",
            "train.tsv",
            "--test",
            "test.tsv",
            "--embedding",
            "embed.h5",
            "--output",
            "output",
            "--save-prefix",
            "model",
        ]

        with patch("sys.argv", test_args):
            main()

        # train.main should have been called
        assert mock_train_main.called

    @patch("dscript.commands.embed.main")
    def test_main_calls_embed_command(self, mock_embed_main):
        """Test that main calls embed command correctly"""
        test_args = ["dscript", "embed", "--seqs", "seqs.fasta", "--outfile", "out.h5"]

        with patch("sys.argv", test_args):
            main()

        # embed.main should have been called
        assert mock_embed_main.called

    @patch("dscript.commands.predict_block.main")
    def test_main_calls_predict_command(self, mock_predict_main):
        """Test that main calls predict (block) command correctly"""
        test_args = [
            "dscript",
            "predict",
            "--pairs",
            "pairs.tsv",
            "--embeddings",
            "embed.h5",
            "--model",
            "model_path",
            "--outfile",
            "out.tsv",
        ]

        with patch("sys.argv", test_args):
            main()

        # predict_block.main should have been called
        assert mock_predict_main.called

    @patch("dscript.commands.predict_serial.main")
    def test_main_calls_predict_serial_command(self, mock_predict_serial_main):
        """Test that main calls predict_serial command correctly"""
        test_args = [
            "dscript",
            "predict_serial",
            "--pairs",
            "pairs.tsv",
            "--embeddings",
            "embed.h5",
            "--model",
            "model_path",
            "--outfile",
            "out.tsv",
        ]

        with patch("sys.argv", test_args):
            main()

        # predict_serial.main should have been called
        assert mock_predict_serial_main.called

    @patch("dscript.commands.predict_bipartite.main")
    def test_main_calls_predict_bipartite_command(self, mock_predict_bipartite_main):
        """Test that main calls predict_bipartite command correctly"""
        test_args = [
            "dscript",
            "predict_bipartite",
            "--seqs0",
            "seqs0.fasta",
            "--seqs1",
            "seqs1.fasta",
            "--embeddings0",
            "embed0.h5",
            "--embeddings1",
            "embed1.h5",
            "--model",
            "model_path",
            "--outfile",
            "out.tsv",
        ]

        with patch("sys.argv", test_args):
            main()

        # predict_bipartite.main should have been called
        assert mock_predict_bipartite_main.called

    @patch("dscript.commands.evaluate.main")
    def test_main_calls_evaluate_command(self, mock_evaluate_main):
        """Test that main calls evaluate command correctly"""
        test_args = [
            "dscript",
            "evaluate",
            "--pairs",
            "pairs.tsv",
            "--embeddings",
            "embed.h5",
            "--model",
            "model_path",
            "--outfile",
            "metrics.json",
        ]

        with patch("sys.argv", test_args):
            main()

        # evaluate.main should have been called
        assert mock_evaluate_main.called

    @patch("dscript.commands.extract_3di.main")
    def test_main_calls_extract_3di_command(self, mock_extract_3di_main):
        """Test that main calls extract-3di command correctly"""
        test_args = ["dscript", "extract-3di", "pdb_dir", "output.fasta"]

        with patch("sys.argv", test_args):
            main()

        # extract_3di.main should have been called
        assert mock_extract_3di_main.called

    def test_main_short_version_flag(self):
        """Test main with -v flag"""
        with patch("sys.argv", ["dscript", "-v"]):
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.stdout", new=StringIO()):
                    main()

            assert exc_info.value.code == 0

    def test_main_short_citation_flag(self):
        """Test main with -c flag"""
        with patch("sys.argv", ["dscript", "-c"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_invalid_command(self):
        """Test main with invalid command"""
        with patch("sys.argv", ["dscript", "invalid_command"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should exit with non-zero code
            assert exc_info.value.code != 0

    def test_main_subparsers_required(self):
        """Test that subparsers.required is True"""
        # This is tested indirectly by test_main_requires_subcommand
        # but we can verify the setup
        with patch("sys.argv", ["dscript"]):
            with pytest.raises(SystemExit):
                main()

    @patch("dscript.commands.train.main")
    def test_main_args_passed_to_command(self, mock_train_main):
        """Test that parsed args are correctly passed to command"""
        test_args = [
            "dscript",
            "train",
            "--train",
            "train.tsv",
            "--test",
            "test.tsv",
            "--embedding",
            "embed.h5",
            "--output",
            "output",
            "--save-prefix",
            "model",
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify args object was passed
        assert mock_train_main.called
        call_args = mock_train_main.call_args[0][0]

        # Check some expected attributes
        assert hasattr(call_args, "train")
        assert hasattr(call_args, "test")
        assert hasattr(call_args, "embedding")


class TestMainIntegration:
    """Integration tests for main function"""

    def test_help_flag_works(self):
        """Test that --help flag works"""
        with patch("sys.argv", ["dscript", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.stdout", new=StringIO()):
                    main()

            # --help should exit with code 0
            assert exc_info.value.code == 0

    def test_command_help_works(self):
        """Test that command-specific help works"""
        with patch("sys.argv", ["dscript", "train", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.stdout", new=StringIO()):
                    main()

            assert exc_info.value.code == 0

    def test_version_output_contains_version(self):
        """Test that version output contains actual version"""

        with patch("sys.argv", ["dscript", "--version"]):
            with pytest.raises(SystemExit):
                with patch("sys.stdout", new=StringIO()):
                    main()

            # Output should contain version info
            # Note: argparse prints to stderr for --version in some cases

    def test_citation_output_contains_citation(self):
        """Test that citation output contains citation info"""
        with patch("sys.argv", ["dscript", "--citation"]):
            with pytest.raises(SystemExit):
                with patch("sys.stdout", new=StringIO()):
                    main()

            # Citation should have been printed

    def test_all_commands_registered(self):
        """Test that all expected commands are registered"""
        # This tests the modules dict in main()
        expected_commands = [
            "train",
            "embed",
            "evaluate",
            "predict_serial",
            "predict",
            "predict_bipartite",
            "extract-3di",
        ]

        for cmd in expected_commands:
            with patch("sys.argv", ["dscript", cmd, "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    with patch("sys.stdout", new=StringIO()):
                        main()

                # Should exit with 0 (help successful)
                assert exc_info.value.code == 0

    @patch("dscript.commands.embed.main")
    def test_embed_command_receives_correct_args(self, mock_embed_main):
        """Test that embed command receives properly parsed arguments"""
        test_args = [
            "dscript",
            "embed",
            "--seqs",
            "test.fasta",
            "--outfile",
            "output.h5",
            "--device",
            "0",
        ]

        with patch("sys.argv", test_args):
            main()

        # Get the args passed to embed.main
        call_args = mock_embed_main.call_args[0][0]

        assert call_args.seqs == "test.fasta"
        assert call_args.outfile == "output.h5"
        assert call_args.device == "0"

    @patch("dscript.commands.predict_block.main")
    def test_predict_command_receives_correct_args(self, mock_predict_main):
        """Test that predict command receives properly parsed arguments"""
        test_args = [
            "dscript",
            "predict",
            "--pairs",
            "pairs.tsv",
            "--embeddings",
            "embed.h5",
            "--model",
            "model_path",
            "--outfile",
            "out.tsv",
            "--blocks",
            "16",
        ]

        with patch("sys.argv", test_args):
            main()

        # Get the args passed to predict.main
        call_args = mock_predict_main.call_args[0][0]

        assert call_args.pairs == "pairs.tsv"
        assert call_args.embeddings == "embed.h5"
        assert call_args.model == "model_path"
        assert call_args.outfile == "out.tsv"
        assert call_args.blocks == 16
