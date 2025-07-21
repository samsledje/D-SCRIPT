import pytest
import torch
import torch.nn as nn

from dscript.models.contact import ContactCNN
from dscript.models.embedding import FullyConnectedEmbed
from dscript.models.interaction import DSCRIPTModel, LogisticActivation, ModelInteraction


class TestLogisticActivation:
    """Test cases for LogisticActivation module."""

    def test_init_default_params(self):
        """Test LogisticActivation initialization with default parameters."""
        activation = LogisticActivation()

        assert activation.x0 == 0
        assert activation.k.item() == 1.0
        assert not activation.k.requires_grad

    def test_init_custom_params(self):
        """Test LogisticActivation initialization with custom parameters."""
        x0 = 0.5
        k = 2.0
        train = True

        activation = LogisticActivation(x0=x0, k=k, train=train)

        assert activation.x0 == x0
        assert activation.k.item() == k
        assert activation.k.requires_grad == train

    def test_forward_shape_preservation(self):
        """Test that forward pass preserves input shape."""
        activation = LogisticActivation()

        test_shapes = [
            (5,),
            (3, 4),
            (2, 3, 4),
            (1, 2, 3, 4),
        ]

        for shape in test_shapes:
            x = torch.randn(shape)
            output = activation(x)
            assert output.shape == shape

    def test_forward_value_range(self):
        """Test that output values are in valid range [0, 1]."""
        activation = LogisticActivation()

        # Test with various input ranges
        test_inputs = [
            torch.randn(10),
            torch.randn(10) * 10,  # Larger values
            torch.randn(10) * -5,  # Negative values
            torch.zeros(5),
            torch.ones(5) * 100,
        ]

        for x in test_inputs:
            output = activation(x)
            assert torch.all(output >= 0)
            assert torch.all(output <= 1)

    def test_forward_sigmoid_behavior(self):
        """Test sigmoid behavior at key points."""
        activation = LogisticActivation(x0=0, k=1)

        # At x0, should be approximately 0.5
        x = torch.tensor([0.0])
        output = activation(x)
        assert torch.allclose(output, torch.tensor([0.5]), atol=1e-6)

        # For large positive values, should approach 1
        x = torch.tensor([10.0])
        output = activation(x)
        assert output.item() > 0.99

        # For large negative values, should approach 0
        x = torch.tensor([-10.0])
        output = activation(x)
        assert output.item() < 0.01

    def test_clip_method(self):
        """Test clip method constrains k to be non-negative."""
        activation = LogisticActivation(k=-2.0, train=True)

        # Initially k might be negative
        assert activation.k.item() == -2.0

        activation.clip()

        # After clipping, k should be non-negative
        assert activation.k.item() >= 0

    def test_custom_midpoint(self):
        """Test sigmoid with custom midpoint."""
        x0 = 2.0
        activation = LogisticActivation(x0=x0, k=1)

        # At custom midpoint, should be approximately 0.5
        x = torch.tensor([x0])
        output = activation(x)
        assert torch.allclose(output, torch.tensor([0.5]), atol=1e-6)


class TestModelInteraction:
    """Test cases for ModelInteraction module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.embed_dim = 100
        self.hidden_dim = 50
        self.embedding = FullyConnectedEmbed(6165, self.embed_dim, 0.5)
        self.contact = ContactCNN(self.embed_dim, self.hidden_dim, 7)

    def test_init_default_params(self):
        """Test ModelInteraction initialization with default parameters."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        assert model.embedding == self.embedding
        assert model.contact == self.contact
        assert not model.use_cuda
        assert model.do_w
        assert model.do_sigmoid
        assert not model.do_pool
        assert model.pool_size == 9

        # Check that parameters are initialized
        assert hasattr(model, "theta")
        assert hasattr(model, "lambda_")
        assert hasattr(model, "gamma")
        assert hasattr(model, "activation")

    def test_init_custom_params(self):
        """Test ModelInteraction initialization with custom parameters."""
        model = ModelInteraction(
            embedding=self.embedding,
            contact=self.contact,
            use_cuda=True,
            do_w=False,
            do_sigmoid=False,
            do_pool=True,
            pool_size=5,
            theta_init=0.5,
            lambda_init=1.0,
            gamma_init=0.1,
        )

        assert model.use_cuda
        assert not model.do_w
        assert not model.do_sigmoid
        assert model.do_pool
        assert model.pool_size == 5

        # When do_w=False, theta and lambda_ should not exist
        assert not hasattr(model, "theta")
        assert not hasattr(model, "lambda_")

        # gamma should still exist
        assert hasattr(model, "gamma")
        assert torch.allclose(model.gamma, torch.ones_like(model.gamma) * 0.1)

    def test_clip_method(self):
        """Test clip method constrains parameters."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        # Set parameters to values that should be clipped
        with torch.no_grad():
            model.theta.fill_(1.5)  # Should be clipped to [0, 1]
            model.lambda_.fill_(-1.0)  # Should be clipped to >= 0
            model.gamma.fill_(-0.5)  # Should be clipped to >= 0

        model.clip()

        assert 0 <= model.theta.item() <= 1
        assert model.lambda_.item() >= 0
        assert model.gamma.item() >= 0

    def test_embed_method_with_embedding(self):
        """Test embed method when embedding model is present."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 2
        seq_len = 10
        input_dim = 6165

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model.embed(x)

        # Should have embedding dimension as output
        expected_shape = (batch_size, seq_len, self.embed_dim)
        assert output.shape == expected_shape

    def test_embed_method_without_embedding(self):
        """Test embed method when embedding model is None."""
        model = ModelInteraction(embedding=None, contact=self.contact, use_cuda=False)

        batch_size = 2
        seq_len = 10
        input_dim = 100

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model.embed(x)

        # Should return input unchanged
        assert torch.equal(output, x)

    def test_cpred_method_basic(self):
        """Test cpred method basic functionality."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len_0 = 8
        seq_len_1 = 10
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        C = model.cpred(z0, z1)

        expected_shape = (batch_size, 1, seq_len_0, seq_len_1)
        assert C.shape == expected_shape

    def test_cpred_method_with_foldseek(self):
        """Test cpred method with foldseek embeddings."""

        batch_size = 1
        seq_len_0 = 8
        seq_len_1 = 10
        input_dim = 6165
        foldseek_dim = 21
        contact = ContactCNN(self.embed_dim + foldseek_dim, self.hidden_dim, 7)

        model = ModelInteraction(
            embedding=self.embedding, contact=contact, use_cuda=False
        )

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)
        f0 = torch.randn(batch_size, seq_len_0, foldseek_dim)
        f1 = torch.randn(batch_size, seq_len_1, foldseek_dim)

        C = model.cpred(z0, z1, embed_foldseek=True, f0=f0, f1=f1)

        expected_shape = (batch_size, 1, seq_len_0, seq_len_1)
        assert C.shape == expected_shape

    def test_predict_method(self):
        """Test predict method returns scalar probability."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        phat = model.predict(z0, z1)

        # Should return a scalar probability
        assert phat.dim() == 0 or phat.shape == torch.Size([1])

        # With sigmoid activation, should be in [0, 1]
        assert 0 <= phat.item() <= 1

    def test_map_predict_method(self):
        """Test map_predict method returns both contact map and probability."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        C, phat = model.map_predict(z0, z1)

        # Contact map shape
        expected_c_shape = (batch_size, 1, seq_len_0, seq_len_1)
        assert C.shape == expected_c_shape

        # Probability should be scalar
        assert phat.dim() == 0 or phat.shape == torch.Size([1])
        assert 0 <= phat.item() <= 1

    def test_forward_method(self):
        """Test forward method (should call predict)."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )
        model = model.eval()

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        output = model(z0, z1)

        # Should behave same as predict
        expected_output = model.predict(z0, z1)
        assert torch.allclose(output, expected_output)

    def test_pooling_behavior(self):
        """Test model with pooling enabled."""
        model = ModelInteraction(
            embedding=self.embedding,
            contact=self.contact,
            use_cuda=False,
            do_pool=True,
            pool_size=3,
        )

        batch_size = 1
        seq_len_0 = 8
        seq_len_1 = 8
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        # Should work with pooling
        phat = model.predict(z0, z1)
        assert 0 <= phat.item() <= 1

    def test_no_sigmoid_behavior(self):
        """Test model without sigmoid activation."""
        model = ModelInteraction(
            embedding=self.embedding,
            contact=self.contact,
            use_cuda=False,
            do_sigmoid=False,
        )

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        phat = model.predict(z0, z1)

        # Without sigmoid, output might be outside [0, 1]
        assert isinstance(phat.item(), float)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim, requires_grad=True)
        z1 = torch.randn(batch_size, seq_len_1, input_dim, requires_grad=True)

        phat = model.predict(z0, z1)
        phat.backward()

        # Check that gradients exist
        assert z0.grad is not None
        assert z1.grad is not None
        assert not torch.allclose(z0.grad, torch.zeros_like(z0.grad))
        assert not torch.allclose(z1.grad, torch.zeros_like(z1.grad))


class TestDSCRIPTModel:
    """Test cases for DSCRIPTModel wrapper."""

    def test_init_default_params(self):
        """Test DSCRIPTModel initialization with default parameters."""
        model = DSCRIPTModel(
            emb_nin=6165,
            emb_nout=100,
            emb_dropout=0.5,
            con_embed_dim=100,
            con_hidden_dim=50,
            con_width=7,
            use_cuda=False,
        )

        assert isinstance(model.embedding, FullyConnectedEmbed)
        assert isinstance(model.contact, ContactCNN)
        assert hasattr(model, "theta")
        assert hasattr(model, "gamma")

    def test_init_custom_params(self):
        """Test DSCRIPTModel initialization with custom parameters."""
        model = DSCRIPTModel(
            emb_nin=1000,
            emb_nout=64,
            emb_dropout=0.3,
            con_embed_dim=64,
            con_hidden_dim=32,
            con_width=5,
            use_cuda=False,
            emb_activation=nn.LeakyReLU(),
            con_activation=nn.ReLU(),
            do_w=False,
            do_sigmoid=False,
            do_pool=True,
            pool_size=5,
            theta_init=0.8,
            lambda_init=0.5,
            gamma_init=0.2,
        )

        assert not model.do_w
        assert not model.do_sigmoid
        assert model.do_pool
        assert model.pool_size == 5

    def test_forward_pass(self):
        """Test complete forward pass through DSCRIPTModel."""
        model = DSCRIPTModel(
            emb_nin=6165,
            emb_nout=100,
            emb_dropout=0.5,
            con_embed_dim=100,
            con_hidden_dim=50,
            con_width=7,
            use_cuda=False,
        )

        batch_size = 1
        seq_len_0 = 8
        seq_len_1 = 10
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        output = model(z0, z1)

        # Should return probability
        assert isinstance(output.item(), float)
        assert 0 <= output.item() <= 1

    def test_inheritance(self):
        """Test that DSCRIPTModel properly inherits from ModelInteraction."""
        model = DSCRIPTModel(
            emb_nin=6165,
            emb_nout=100,
            emb_dropout=0.5,
            con_embed_dim=100,
            con_hidden_dim=50,
            con_width=7,
            use_cuda=False,
        )

        # Should have ModelInteraction methods
        assert hasattr(model, "predict")
        assert hasattr(model, "map_predict")
        assert hasattr(model, "cpred")
        assert hasattr(model, "embed")
        assert hasattr(model, "clip")

        # Should be instance of ModelInteraction
        assert isinstance(model, ModelInteraction)


class TestModelInteractionEdgeCases:
    """Test edge cases and error conditions for ModelInteraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.embed_dim = 100
        self.hidden_dim = 50
        self.embedding = FullyConnectedEmbed(6165, self.embed_dim, 0.5)
        self.contact = ContactCNN(self.embed_dim, self.hidden_dim, 7)

    def test_foldseek_assertion_errors(self):
        """Test assertion errors in foldseek embedding."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        # Test with embed_foldseek=True but missing f0/f1
        with pytest.raises(AssertionError):
            model.cpred(z0, z1, embed_foldseek=True, f0=None, f1=None)

        # Test with wrong types
        with pytest.raises(AssertionError):
            model.cpred(z0, z1, embed_foldseek=True, f0="not_tensor", f1=None)

    def test_very_small_sequences(self):
        """Test with very small sequence lengths."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len = 2
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len, input_dim)
        z1 = torch.randn(batch_size, seq_len, input_dim)

        # Should handle single residue sequences
        phat = model.predict(z0, z1)
        assert isinstance(phat.item(), float)

    def test_large_sequences(self):
        """Test with larger sequence lengths."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        batch_size = 1
        seq_len_0 = 800
        seq_len_1 = 900
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        # Should handle larger sequences (might be slow but should work)
        phat = model.predict(z0, z1)
        assert isinstance(phat.item(), float)

    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = ModelInteraction(
            embedding=self.embedding, contact=self.contact, use_cuda=False
        )

        model.eval()

        batch_size = 1
        seq_len_0 = 5
        seq_len_1 = 6
        input_dim = 6165

        z0 = torch.randn(batch_size, seq_len_0, input_dim)
        z1 = torch.randn(batch_size, seq_len_1, input_dim)

        with torch.no_grad():
            phat = model.predict(z0, z1)

        assert isinstance(phat.item(), float)
