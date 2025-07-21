import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from dscript.models.embedding import (
    FullyConnectedEmbed,
    IdentityEmbed,
    SkipLSTM,
)


class TestIdentityEmbed:
    """Test cases for IdentityEmbed model."""

    def test_identity_embed_initialization(self):
        """Test IdentityEmbed initialization."""
        model = IdentityEmbed()
        assert isinstance(model, nn.Module)

    def test_identity_embed_forward_pass(self):
        """Test IdentityEmbed forward pass."""
        model = IdentityEmbed()

        # Test with various input shapes
        input_2d = torch.randn(10, 20)
        output_2d = model(input_2d)
        assert torch.equal(input_2d, output_2d)

        input_3d = torch.randn(5, 10, 15)
        output_3d = model(input_3d)
        assert torch.equal(input_3d, output_3d)

        input_4d = torch.randn(2, 5, 10, 15)
        output_4d = model(input_4d)
        assert torch.equal(input_4d, output_4d)

    def test_identity_embed_gradient_flow(self):
        """Test gradient flow through IdentityEmbed."""
        model = IdentityEmbed()

        input_tensor = torch.randn(5, 10, requires_grad=True)
        output = model(input_tensor)

        loss = torch.sum(output)
        loss.backward()

        # Gradients should flow through unchanged
        assert input_tensor.grad is not None
        assert torch.equal(input_tensor.grad, torch.ones_like(input_tensor))

    def test_identity_embed_device_compatibility(self):
        """Test device compatibility."""
        model = IdentityEmbed()

        # Test CPU
        input_cpu = torch.randn(5, 10)
        output_cpu = model(input_cpu)
        assert output_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            input_cuda = torch.randn(5, 10).cuda()
            output_cuda = model_cuda(input_cuda)
            assert output_cuda.device.type == "cuda"


class TestFullyConnectedEmbed:
    """Test cases for FullyConnectedEmbed model."""

    def test_fully_connected_embed_initialization(self):
        """Test FullyConnectedEmbed initialization."""
        nin = 1024
        nout = 100
        dropout = 0.3

        model = FullyConnectedEmbed(nin, nout, dropout)

        assert model.nin == nin
        assert model.nout == nout
        assert model.dropout_p == dropout
        assert isinstance(model.transform, nn.Linear)
        assert isinstance(model.drop, nn.Dropout)
        assert isinstance(model.activation, nn.Module)

        # Check linear layer dimensions
        assert model.transform.in_features == nin
        assert model.transform.out_features == nout
        assert model.drop.p == dropout

    def test_fully_connected_embed_custom_activation(self):
        """Test FullyConnectedEmbed with custom activation."""
        model = FullyConnectedEmbed(100, 50, activation=nn.Tanh())
        assert isinstance(model.activation, nn.Tanh)

        model = FullyConnectedEmbed(100, 50, activation=nn.Sigmoid())
        assert isinstance(model.activation, nn.Sigmoid)

    def test_fully_connected_embed_forward_pass(self):
        """Test FullyConnectedEmbed forward pass."""
        nin = 1024
        nout = 100
        batch_size = 5
        seq_length = 20

        model = FullyConnectedEmbed(nin, nout)

        # Test 3D input (batch_size, seq_length, embedding_dim)
        input_tensor = torch.randn(batch_size, seq_length, nin)
        output = model(input_tensor)

        assert output.shape == (batch_size, seq_length, nout)
        assert torch.all(torch.isfinite(output))

    def test_fully_connected_embed_different_shapes(self):
        """Test FullyConnectedEmbed with different input shapes."""
        model = FullyConnectedEmbed(50, 25)

        # Test 2D input
        input_2d = torch.randn(10, 50)
        output_2d = model(input_2d)
        assert output_2d.shape == (10, 25)

        # Test 3D input
        input_3d = torch.randn(5, 15, 50)
        output_3d = model(input_3d)
        assert output_3d.shape == (5, 15, 25)

        # Test 4D input
        input_4d = torch.randn(2, 3, 10, 50)
        output_4d = model(input_4d)
        assert output_4d.shape == (2, 3, 10, 25)

    def test_fully_connected_embed_dropout_effect(self):
        """Test dropout effect during training vs evaluation."""
        model = FullyConnectedEmbed(100, 50, dropout=0.5)
        input_tensor = torch.randn(10, 100)

        # Training mode
        model.train()
        output_train = model(input_tensor)

        # Evaluation mode
        model.eval()
        output_eval = model(input_tensor)

        # Outputs should have same shape
        assert output_train.shape == output_eval.shape
        assert output_train.shape == (10, 50)

    def test_fully_connected_embed_gradient_flow(self):
        """Test gradient flow through FullyConnectedEmbed."""
        model = FullyConnectedEmbed(50, 25)
        input_tensor = torch.randn(5, 50, requires_grad=True)

        output = model(input_tensor)
        loss = torch.sum(output)
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert model.transform.weight.grad is not None
        assert model.transform.bias.grad is not None

        # Check gradients are not all zeros
        assert not torch.all(input_tensor.grad == 0)
        assert not torch.all(model.transform.weight.grad == 0)

    def test_fully_connected_embed_zero_dropout(self):
        """Test FullyConnectedEmbed with zero dropout."""
        model = FullyConnectedEmbed(100, 50, dropout=0.0)
        assert model.drop.p == 0.0

        input_tensor = torch.randn(5, 100)

        # With zero dropout, outputs should be deterministic
        model.eval()
        output1 = model(input_tensor)
        output2 = model(input_tensor)

        assert torch.allclose(output1, output2)


class TestSkipLSTM:
    """Test cases for SkipLSTM model."""

    def test_skip_lstm_initialization(self):
        """Test SkipLSTM initialization."""
        nin = 21
        nout = 100
        hidden_dim = 512
        num_layers = 3

        model = SkipLSTM(nin, nout, hidden_dim, num_layers)

        assert model.nin == nin
        assert model.nout == nout
        assert isinstance(model.dropout, nn.Dropout)
        assert len(model.layers) == num_layers
        assert isinstance(model.proj, nn.Linear)

        # Check LSTM layers
        for layer in model.layers:
            assert isinstance(layer, nn.LSTM)
            assert layer.batch_first
            assert layer.bidirectional

    def test_skip_lstm_custom_parameters(self):
        """Test SkipLSTM with custom parameters."""
        model = SkipLSTM(
            nin=25,
            nout=200,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
        )

        assert model.nin == 25
        assert model.nout == 200
        assert len(model.layers) == 2

        # Check bidirectional setting
        for layer in model.layers:
            assert not layer.bidirectional

    def test_skip_lstm_projection_layer_size(self):
        """Test SkipLSTM projection layer has correct size."""
        nin = 21
        nout = 100
        hidden_dim = 512
        num_layers = 3

        # Bidirectional case
        model_bi = SkipLSTM(nin, nout, hidden_dim, num_layers, bidirectional=True)
        expected_proj_in_bi = 2 * hidden_dim * num_layers + nin
        assert model_bi.proj.in_features == expected_proj_in_bi
        assert model_bi.proj.out_features == nout

        # Unidirectional case
        model_uni = SkipLSTM(nin, nout, hidden_dim, num_layers, bidirectional=False)
        expected_proj_in_uni = hidden_dim * num_layers + nin
        assert model_uni.proj.in_features == expected_proj_in_uni
        assert model_uni.proj.out_features == nout

    def test_skip_lstm_to_one_hot_tensor(self):
        """Test to_one_hot method with regular tensors."""
        nin = 21
        model = SkipLSTM(nin, 100, 512, 3)

        # Test 2D input (batch_size, seq_length)
        input_tensor = torch.randint(0, nin, (5, 10))
        one_hot = model.to_one_hot(input_tensor)

        assert one_hot.shape == (5, 10, nin)
        assert torch.all((one_hot == 0) | (one_hot == 1))  # Should be binary

        # Check that each position has exactly one 1
        assert torch.all(torch.sum(one_hot, dim=2) == 1)

    def test_skip_lstm_to_one_hot_packed_sequence(self):
        """Test to_one_hot method with PackedSequence."""
        nin = 21
        model = SkipLSTM(nin, 100, 512, 3)

        # Create a PackedSequence
        sequences = [
            torch.randint(0, nin, (10,)),
            torch.randint(0, nin, (8,)),
            torch.randint(0, nin, (6,)),
        ]
        lengths = [10, 8, 6]
        packed_input = pack_padded_sequence(
            torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True),
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        one_hot_packed = model.to_one_hot(packed_input)

        assert isinstance(one_hot_packed, PackedSequence)
        assert one_hot_packed.data.shape[1] == nin
        assert torch.all((one_hot_packed.data == 0) | (one_hot_packed.data == 1))

    def test_skip_lstm_transform_method(self):
        """Test transform method."""
        nin = 21
        nout = 100
        hidden_dim = 256
        num_layers = 2

        model = SkipLSTM(nin, nout, hidden_dim, num_layers)

        # Test with regular tensor
        input_tensor = torch.randint(0, nin, (3, 15))
        transformed = model.transform(input_tensor)

        expected_dim = nin + 2 * hidden_dim * num_layers  # bidirectional
        assert transformed.shape == (3, 15, expected_dim)

    def test_skip_lstm_forward_pass_tensor(self):
        """Test forward pass with regular tensors."""
        nin = 21
        nout = 100
        hidden_dim = 256
        num_layers = 2
        batch_size = 5
        seq_length = 12

        model = SkipLSTM(nin, nout, hidden_dim, num_layers)

        input_tensor = torch.randint(0, nin, (batch_size, seq_length))
        output = model(input_tensor)

        assert output.shape == (batch_size, seq_length, nout)
        assert torch.all(torch.isfinite(output))

    def test_skip_lstm_forward_pass_packed_sequence(self):
        """Test forward pass with PackedSequence."""
        nin = 21
        nout = 100
        hidden_dim = 256
        num_layers = 2

        model = SkipLSTM(nin, nout, hidden_dim, num_layers)

        # Create PackedSequence with variable length sequences
        sequences = [
            torch.randint(0, nin, (15,)),
            torch.randint(0, nin, (10,)),
            torch.randint(0, nin, (8,)),
        ]
        lengths = [15, 10, 8]
        packed_input = pack_padded_sequence(
            torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True),
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        output = model(packed_input)

        assert isinstance(output, PackedSequence)
        assert output.data.shape[1] == nout
        assert torch.all(torch.isfinite(output.data))

    def test_skip_lstm_different_sequence_lengths(self):
        """Test SkipLSTM with different sequence lengths."""
        model = SkipLSTM(21, 100, 256, 2)

        # Test various sequence lengths
        for seq_len in [5, 10, 20, 50]:
            input_tensor = torch.randint(0, 21, (2, seq_len))
            output = model(input_tensor)
            assert output.shape == (2, seq_len, 100)

    def test_skip_lstm_layer_dimensions(self):
        """Test that LSTM layers have correct dimensions."""
        nin = 21
        hidden_dim = 256
        num_layers = 3

        model = SkipLSTM(nin, 100, hidden_dim, num_layers, bidirectional=True)

        # First layer should take nin inputs
        assert model.layers[0].input_size == nin
        assert model.layers[0].hidden_size == hidden_dim

        # Subsequent layers should take 2*hidden_dim inputs (bidirectional)
        for i in range(1, num_layers):
            assert model.layers[i].input_size == 2 * hidden_dim
            assert model.layers[i].hidden_size == hidden_dim

    def test_skip_lstm_unidirectional_dimensions(self):
        """Test SkipLSTM with unidirectional LSTM."""
        nin = 21
        hidden_dim = 256
        num_layers = 3

        model = SkipLSTM(nin, 100, hidden_dim, num_layers, bidirectional=False)

        # First layer should take nin inputs
        assert model.layers[0].input_size == nin

        # Subsequent layers should take hidden_dim inputs (unidirectional)
        for i in range(1, num_layers):
            assert model.layers[i].input_size == hidden_dim

    def test_skip_lstm_out_of_range_input(self):
        """Test SkipLSTM behavior with out of range inputs."""
        nin = 21
        model = SkipLSTM(nin, 100, 256, 2)

        # Input with values >= nin should cause issues in one-hot encoding
        input_tensor = torch.randint(0, nin + 5, (2, 10))  # Some values >= nin

        # This should not crash but may produce unexpected results
        with torch.no_grad():
            try:
                output = model(input_tensor)
                # If it doesn't crash, check output shape
                assert output.shape == (2, 10, 100)
            except (RuntimeError, IndexError):
                # Expected for out of range indices
                pass

    def test_skip_lstm_single_element_sequence(self):
        """Test SkipLSTM with single element sequences."""
        model = SkipLSTM(21, 100, 256, 2)

        input_tensor = torch.randint(0, 21, (3, 1))
        output = model(input_tensor)

        assert output.shape == (3, 1, 100)
        assert torch.all(torch.isfinite(output))


class TestEmbeddingModelsIntegration:
    """Integration tests for embedding models."""

    def test_models_can_be_combined(self):
        """Test that embedding models can be used together."""
        # Test chaining FullyConnectedEmbed after SkipLSTM
        skip_lstm = SkipLSTM(21, 200, 256, 2)
        fc_embed = FullyConnectedEmbed(200, 100)

        input_tensor = torch.randint(0, 21, (3, 15))

        # Forward through SkipLSTM first
        intermediate = skip_lstm(input_tensor)
        assert intermediate.shape == (3, 15, 200)

        # Then through FullyConnectedEmbed
        final_output = fc_embed(intermediate)
        assert final_output.shape == (3, 15, 100)

    def test_models_device_consistency(self):
        """Test that all models can be moved to same device."""
        identity = IdentityEmbed()
        fc_embed = FullyConnectedEmbed(100, 50)
        skip_lstm = SkipLSTM(21, 100, 128, 2)

        models = [identity, fc_embed, skip_lstm]

        # Test CPU
        for model in models:
            model.cpu()
            if len(list(model.parameters())) > 0:
                assert next(model.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            for model in models:
                model.cuda()
                if len(list(model.parameters())) > 0:
                    assert next(model.parameters()).device.type == "cuda"

    def test_models_state_dict_save_load(self):
        """Test saving and loading model state dicts."""
        # Test FullyConnectedEmbed
        fc_model = FullyConnectedEmbed(100, 50)
        fc_state = fc_model.state_dict()

        fc_model_new = FullyConnectedEmbed(100, 50)
        fc_model_new.load_state_dict(fc_state)

        # Test that weights are the same
        assert torch.allclose(fc_model.transform.weight, fc_model_new.transform.weight)
        assert torch.allclose(fc_model.transform.bias, fc_model_new.transform.bias)

        # Test SkipLSTM
        skip_model = SkipLSTM(21, 100, 128, 2)
        skip_state = skip_model.state_dict()

        skip_model_new = SkipLSTM(21, 100, 128, 2)
        skip_model_new.load_state_dict(skip_state)

        # Test one parameter to verify loading worked
        assert torch.allclose(skip_model.proj.weight, skip_model_new.proj.weight)
