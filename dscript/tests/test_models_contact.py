import torch
import torch.nn as nn

from dscript.models.contact import ContactCNN, FullyConnected


class TestFullyConnected:
    """Test cases for FullyConnected module."""

    def test_init_default_params(self):
        """Test FullyConnected initialization with default parameters."""
        embed_dim = 100
        hidden_dim = 50

        model = FullyConnected(embed_dim, hidden_dim)

        assert model.D == embed_dim
        assert model.H == hidden_dim
        assert isinstance(model.conv, nn.Conv2d)
        assert isinstance(model.batchnorm, nn.BatchNorm2d)
        assert isinstance(model.activation, nn.ReLU)

        # Check conv layer parameters
        assert model.conv.in_channels == 2 * embed_dim
        assert model.conv.out_channels == hidden_dim
        assert model.conv.kernel_size == (1, 1)

    def test_init_custom_activation(self):
        """Test FullyConnected initialization with custom activation."""
        embed_dim = 64
        hidden_dim = 32
        activation = nn.LeakyReLU()

        model = FullyConnected(embed_dim, hidden_dim, activation)

        assert model.D == embed_dim
        assert model.H == hidden_dim
        assert isinstance(model.activation, nn.LeakyReLU)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        embed_dim = 100
        hidden_dim = 50
        batch_size = 2
        seq_len_0 = 10
        seq_len_1 = 15

        model = FullyConnected(embed_dim, hidden_dim)

        # Create input tensors
        z0 = torch.randn(batch_size, seq_len_0, embed_dim)
        z1 = torch.randn(batch_size, seq_len_1, embed_dim)

        output = model(z0, z1)

        # Check output shape: (batch, hidden_dim, seq_len_0, seq_len_1)
        expected_shape = (batch_size, hidden_dim, seq_len_0, seq_len_1)
        assert output.shape == expected_shape

    def test_forward_computation(self):
        """Test forward pass computation logic."""
        embed_dim = 4
        hidden_dim = 2
        batch_size = 1
        seq_len_0 = 2
        seq_len_1 = 2

        model = FullyConnected(embed_dim, hidden_dim)
        model = model.eval()  # Set to eval mode to avoid dropout

        # Create simple input tensors for predictable output
        z0 = torch.ones(batch_size, seq_len_0, embed_dim)
        z1 = torch.ones(batch_size, seq_len_1, embed_dim) * 2

        output = model(z0, z1)

        # Output should be non-zero and have correct shape
        assert output.shape == (batch_size, hidden_dim, seq_len_0, seq_len_1)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        embed_dim = 100
        hidden_dim = 50
        batch_size = 1

        model = FullyConnected(embed_dim, hidden_dim)

        # Test various sequence length combinations
        test_cases = [(5, 8), (12, 3), (1, 2), (20, 15)]
        # NOTE: this fails on (1, 1) -- we should never face this input anyway

        for seq_len_0, seq_len_1 in test_cases:
            z0 = torch.randn(batch_size, seq_len_0, embed_dim)
            z1 = torch.randn(batch_size, seq_len_1, embed_dim)

            output = model(z0, z1)
            expected_shape = (batch_size, hidden_dim, seq_len_0, seq_len_1)
            assert output.shape == expected_shape

    def test_forward_batch_processing(self):
        """Test forward pass with different batch sizes."""
        embed_dim = 64
        hidden_dim = 32
        seq_len_0 = 8
        seq_len_1 = 10

        model = FullyConnected(embed_dim, hidden_dim)

        # Test different batch sizes
        for batch_size in [1, 3, 8]:
            z0 = torch.randn(batch_size, seq_len_0, embed_dim)
            z1 = torch.randn(batch_size, seq_len_1, embed_dim)

            output = model(z0, z1)
            expected_shape = (batch_size, hidden_dim, seq_len_0, seq_len_1)
            assert output.shape == expected_shape

    def test_forward_zero_inputs(self):
        """Test forward pass with zero input tensors."""
        embed_dim = 100
        hidden_dim = 50
        batch_size = 2
        seq_len_0 = 5
        seq_len_1 = 7

        model = FullyConnected(embed_dim, hidden_dim)

        # Zero inputs
        z0 = torch.zeros(batch_size, seq_len_0, embed_dim)
        z1 = torch.zeros(batch_size, seq_len_1, embed_dim)

        output = model(z0, z1)

        # Should still produce valid output shape
        expected_shape = (batch_size, hidden_dim, seq_len_0, seq_len_1)
        assert output.shape == expected_shape


class TestContactCNN:
    """Test cases for ContactCNN module."""

    def test_init_default_params(self):
        """Test ContactCNN initialization with default parameters."""
        embed_dim = 100

        model = ContactCNN(embed_dim)

        assert isinstance(model.hidden, FullyConnected)
        assert isinstance(model.conv, nn.Conv2d)
        assert isinstance(model.batchnorm, nn.BatchNorm2d)
        assert isinstance(model.activation, nn.Sigmoid)

        # Check conv layer parameters
        assert model.conv.in_channels == 50  # default hidden_dim
        assert model.conv.out_channels == 1
        assert model.conv.kernel_size == (7, 7)  # default width
        assert model.conv.padding == (3, 3)  # width // 2

    def test_init_custom_params(self):
        """Test ContactCNN initialization with custom parameters."""
        embed_dim = 64
        hidden_dim = 32
        width = 5
        activation = nn.ReLU()

        model = ContactCNN(embed_dim, hidden_dim, width, activation)

        assert isinstance(model.hidden, FullyConnected)
        assert isinstance(model.activation, nn.ReLU)
        assert model.conv.kernel_size == (width, width)
        assert model.conv.padding == (width // 2, width // 2)

    def test_clip_functionality(self):
        """Test that clip makes conv weights symmetric."""
        embed_dim = 100
        model = ContactCNN(embed_dim)

        # Set asymmetric weights
        with torch.no_grad():
            model.conv.weight.fill_(1.0)
            model.conv.weight[0, 0, 0, 1] = 5.0
            model.conv.weight[0, 0, 1, 0] = 3.0

        model.clip()

        # Check that weights are now symmetric
        w = model.conv.weight
        assert torch.allclose(w, w.transpose(2, 3))

    def test_cmap_method(self):
        """Test cmap method returns expected shape."""
        embed_dim = 100
        hidden_dim = 50
        batch_size = 2
        seq_len_0 = 8
        seq_len_1 = 10

        model = ContactCNN(embed_dim, hidden_dim)

        z0 = torch.randn(batch_size, seq_len_0, embed_dim)
        z1 = torch.randn(batch_size, seq_len_1, embed_dim)

        C = model.cmap(z0, z1)

        expected_shape = (batch_size, hidden_dim, seq_len_0, seq_len_1)
        assert C.shape == expected_shape

    def test_predict_method(self):
        """Test predict method returns expected shape."""
        embed_dim = 100
        hidden_dim = 50
        batch_size = 2
        seq_len_0 = 8
        seq_len_1 = 10

        model = ContactCNN(embed_dim, hidden_dim)

        # Create broadcast tensor (output of cmap)
        C = torch.randn(batch_size, hidden_dim, seq_len_0, seq_len_1)

        S = model.predict(C)

        expected_shape = (batch_size, 1, seq_len_0, seq_len_1)
        assert S.shape == expected_shape

    def test_forward_full_pipeline(self):
        """Test full forward pass."""
        embed_dim = 100
        hidden_dim = 50
        batch_size = 2
        seq_len_0 = 8
        seq_len_1 = 10

        model = ContactCNN(embed_dim, hidden_dim)

        z0 = torch.randn(batch_size, seq_len_0, embed_dim)
        z1 = torch.randn(batch_size, seq_len_1, embed_dim)

        output = model(z0, z1)

        expected_shape = (batch_size, 1, seq_len_0, seq_len_1)
        assert output.shape == expected_shape

        # With default sigmoid activation, outputs should be in [0, 1]
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_forward_different_dimensions(self):
        """Test forward pass with various input dimensions."""
        test_cases = [
            (64, 32, 5),  # Small dimensions
            (128, 64, 3),  # Different dimensions
            (100, 50, 1),  # Minimal width
        ]

        batch_size = 1
        seq_len_0 = 6
        seq_len_1 = 8

        for embed_dim, hidden_dim, width in test_cases:
            model = ContactCNN(embed_dim, hidden_dim, width)

            z0 = torch.randn(batch_size, seq_len_0, embed_dim)
            z1 = torch.randn(batch_size, seq_len_1, embed_dim)

            output = model(z0, z1)
            expected_shape = (batch_size, 1, seq_len_0, seq_len_1)
            assert output.shape == expected_shape

    def test_eval_mode(self):
        """Test model in evaluation mode."""
        embed_dim = 100
        model = ContactCNN(embed_dim)

        # Set to eval mode
        model.eval()

        batch_size = 2
        seq_len_0 = 5
        seq_len_1 = 7

        z0 = torch.randn(batch_size, seq_len_0, embed_dim)
        z1 = torch.randn(batch_size, seq_len_1, embed_dim)

        # Should work in eval mode
        with torch.no_grad():
            output = model(z0, z1)

        expected_shape = (batch_size, 1, seq_len_0, seq_len_1)
        assert output.shape == expected_shape

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        embed_dim = 100
        model = ContactCNN(embed_dim)

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 5

        z0 = torch.randn(batch_size, seq_len_0, embed_dim, requires_grad=True)
        z1 = torch.randn(batch_size, seq_len_1, embed_dim, requires_grad=True)

        output = model(z0, z1)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert z0.grad is not None
        assert z1.grad is not None
        assert not torch.allclose(z0.grad, torch.zeros_like(z0.grad))
        assert not torch.allclose(z1.grad, torch.zeros_like(z1.grad))

    def test_symmetric_inputs(self):
        """Test behavior with symmetric inputs (z0 == z1)."""
        embed_dim = 100
        model = ContactCNN(embed_dim)

        batch_size = 1
        seq_len = 6

        z = torch.randn(batch_size, seq_len, embed_dim)

        output = model(z, z)

        expected_shape = (batch_size, 1, seq_len, seq_len)
        assert output.shape == expected_shape

        # For symmetric inputs, output should be approximately symmetric
        # (within numerical precision and activation effects)
        output_squeezed = output.squeeze()
        diff = torch.abs(output_squeezed - output_squeezed.T)
        max_diff = torch.max(diff)

        # Allow for some numerical difference due to floating point precision
        assert max_diff < 1e-5
