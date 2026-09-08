"""
Property-based tests for VectorField.

Feature: matching-flow-tkg
Property 3: VectorField output dimension invariant

For arbitrary batch sizes and d_model values, assert that
VectorField(d_model, dropout).forward(x_t, t_emb, ctx).shape == (bs, d_model).

Validates: Requirements 4.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from models.MatchingFlow import VectorField


# Strategy: sample d_model as multiples of 1 in a reasonable range,
# batch size from 1 to 32, dropout between 0.0 and 0.5.
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    d_model=st.integers(min_value=4, max_value=128),
    bs=st.integers(min_value=1, max_value=32),
    dropout=st.floats(min_value=0.0, max_value=0.5),
)
def test_property_3_vector_field_output_dimension_invariant(d_model, bs, dropout):
    """
    **Property 3: VectorField output dimension invariant**

    For any valid batch size and d_model, VectorField.forward must return
    a tensor of shape (bs, d_model).

    Validates: Requirements 4.1
    """
    model = VectorField(d_model=d_model, dropout=dropout)
    model.eval()  # disable dropout for deterministic shape check

    x_t = torch.randn(bs, d_model)
    time_emb = torch.randn(bs, d_model)
    context_emb = torch.randn(bs, d_model)

    with torch.no_grad():
        output = model(x_t, time_emb, context_emb)

    assert output.shape == (bs, d_model), (
        f"Expected output shape ({bs}, {d_model}), got {tuple(output.shape)}"
    )
