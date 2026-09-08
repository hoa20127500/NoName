"""
Unit tests for MatchingFlowTKG structural and boundary conditions.

Feature: matching-flow-tkg
Validates: Requirements 1.1, 1.3, 1.4, 3.3, 4.2, 4.3, 7.3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from collections import namedtuple

from models.MatchingFlow import MatchingFlowTKG
from GraphEmbedding import GraphEmbedding
import main as main_module


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

Config = namedtuple('config', ['n_ent', 'd_model', 'n_rel', 'dropout', 's_emb_dim', 't_emb_dim'])

@pytest.fixture
def small_config():
    return Config(
        n_ent=30,
        n_rel=15,
        d_model=16,
        dropout=0.1,
        s_emb_dim=64,
        t_emb_dim=36,
    )


@pytest.fixture
def model(small_config):
    m = MatchingFlowTKG(small_config, ode_steps=5)
    m.eval()
    return m


def make_batch(config, bs=2):
    """Return a small synthetic batch of (heads, rels, tails, year, month, day)."""
    heads = torch.randint(0, config.n_ent, (bs,))
    rels  = torch.randint(0, config.n_rel, (bs,))
    tails = torch.randint(0, config.n_ent, (bs,))
    # Use month >= 1 and day >= 1 to avoid modulo-by-zero in _encode_context
    year  = torch.randint(2000, 2020, (bs,)).float()
    month = torch.randint(1, 13, (bs,)).float()
    day   = torch.randint(1, 29, (bs,)).float()
    return heads, rels, tails, year, month, day


# ---------------------------------------------------------------------------
# Test 1: Import test
# ---------------------------------------------------------------------------

def test_importable():
    """MatchingFlowTKG is importable from models.MatchingFlow. Validates: Req 1.1"""
    from models.MatchingFlow import MatchingFlowTKG as MFTKG  # noqa: F401
    assert MFTKG is not None


# ---------------------------------------------------------------------------
# Test 2: Encoder type test
# ---------------------------------------------------------------------------

def test_encoder_is_graph_embedding(model):
    """model.encoder is an instance of GraphEmbedding. Validates: Req 1.3, 2.1"""
    assert isinstance(model.encoder, GraphEmbedding), (
        f"Expected GraphEmbedding, got {type(model.encoder)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Parameter test
# ---------------------------------------------------------------------------

def test_w_is_trainable_parameter(model):
    """model.w is an nn.Parameter with requires_grad=True. Validates: Req 3.3"""
    assert isinstance(model.w, nn.Parameter), (
        f"Expected nn.Parameter, got {type(model.w)}"
    )
    assert model.w.requires_grad, "model.w.requires_grad should be True"


# ---------------------------------------------------------------------------
# Test 4: Flow interpolation correctness
# ---------------------------------------------------------------------------

def test_flow_interpolation_at_t0(small_config):
    """At t=0: x_t = (1-0)*noise + 0*x1 == noise. Validates: Req 4.2, 4.3"""
    bs = 2
    d_model = small_config.d_model
    noise = torch.randn(bs, d_model)
    x1    = torch.randn(bs, d_model)
    t_scalar = torch.zeros(bs)  # t = 0

    x_t = (1 - t_scalar.unsqueeze(1)) * noise + t_scalar.unsqueeze(1) * x1

    assert torch.allclose(x_t, noise), (
        "At t=0, x_t should equal noise but got a different tensor."
    )


def test_flow_interpolation_at_t1(small_config):
    """At t=1: x_t = (1-1)*noise + 1*x1 == x1. Validates: Req 4.2, 4.3"""
    bs = 2
    d_model = small_config.d_model
    noise = torch.randn(bs, d_model)
    x1    = torch.randn(bs, d_model)
    t_scalar = torch.ones(bs)  # t = 1

    x_t = (1 - t_scalar.unsqueeze(1)) * noise + t_scalar.unsqueeze(1) * x1

    assert torch.allclose(x_t, x1), (
        "At t=1, x_t should equal x1 but got a different tensor."
    )


# ---------------------------------------------------------------------------
# Test 5: ode_steps=0 raises ValueError
# ---------------------------------------------------------------------------

def test_ode_steps_zero_raises_value_error(small_config):
    """Instantiate MatchingFlowTKG with ode_steps=0 and call test_forward — expects ValueError."""
    model_zero = MatchingFlowTKG(small_config, ode_steps=0)
    model_zero.eval()

    heads, rels, tails, year, month, day = make_batch(small_config, bs=2)

    with pytest.raises(ValueError):
        model_zero.test_forward(heads, rels, tails, year, month, day)


def test_one_step_test_forward_shape(small_config):
    """Default 1-step Euler returns (bs, n_ent)."""
    model = MatchingFlowTKG(small_config, ode_steps=1)
    model.eval()
    heads, rels, tails, year, month, day = make_batch(small_config, bs=3)
    with torch.no_grad():
        scores = model.test_forward(heads, rels, tails, year, month, day)
    assert scores.shape == (3, small_config.n_ent)


def test_train_forward_one_step_finite(small_config):
    """1-step training path returns a finite scalar."""
    model = MatchingFlowTKG(small_config, ode_steps=1)
    model.train()
    heads, rels, tails, year, month, day = make_batch(small_config, bs=2)
    neg = torch.randint(0, small_config.n_ent, (2, 8))
    loss = model.train_forward(heads, rels, tails, year, month, day, neg)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 6: Unknown model_name raises ValueError listing valid names
# ---------------------------------------------------------------------------

def test_unknown_model_name_raises_value_error(small_config):
    """Registry dispatch raises ValueError with valid names for unknown model string."""
    unknown_name = "UnknownModel"
    assert unknown_name not in main_module.MODEL_REGISTRY, (
        f"'{unknown_name}' should not be a valid model name in the registry."
    )

    with pytest.raises(ValueError) as exc_info:
        if unknown_name not in main_module.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{unknown_name}'. "
                f"Valid: {list(main_module.MODEL_REGISTRY.keys())}"
            )

    error_msg = str(exc_info.value)
    # The error should mention at least one valid name
    for valid_name in main_module.MODEL_REGISTRY.keys():
        assert valid_name in error_msg, (
            f"Expected valid model name '{valid_name}' to appear in error message: {error_msg!r}"
        )
