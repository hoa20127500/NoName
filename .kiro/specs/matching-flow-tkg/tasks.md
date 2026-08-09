# Implementation Plan: Matching Flow TKG (`matching-flow-tkg`)

## Overview

Implement `MatchingFlowTKG`, a conditional flow-matching model for temporal knowledge graph link prediction, as a drop-in alternative to the existing `NoName` diffusion model. The implementation lives in a new file `models/MatchingFlow.py` and requires a minimal, isolated change to `main.py` for model selection. No existing files (`model.py`, `Denoiser.py`, `GraphEmbedding.py`, `dataset.py`) are modified.

## Tasks

- [x] 1. Create `models/MatchingFlow.py` with the `VectorField` MLP
  - Create the file `models/MatchingFlow.py` with necessary imports (`torch`, `torch.nn`, `torch.nn.functional`, `math`, `GraphEmbedding`, `regularizers.N3`).
  - Define `VectorField(nn.Module)` with:
    - `__init__(self, d_model, dropout)`: builds the three-layer MLP (`Linear(3*d_model→4*d_model)→LayerNorm→ReLU→Dropout`, `Linear(4*d_model→2*d_model)→LayerNorm→ReLU→Dropout`, `Linear(2*d_model→d_model)`) plus a skip connection `Linear(3*d_model→d_model)`.
    - `forward(self, x_t, time_emb, context_emb) -> Tensor`: concatenates the three inputs along dim=-1, runs through MLP layers, adds skip, returns `(bs, d_model)` velocity vector.
  - _Requirements: 4.1, 4.5_

  - [ ]* 1.1 Write property test for VectorField output dimension invariant
    - **Property 3: VectorField output dimension invariant**
    - For arbitrary batch sizes and `d_model` values, assert `VectorField(d_model, dropout).forward(x_t, t_emb, ctx).shape == (bs, d_model)`.
    - Use `hypothesis` with `@settings(max_examples=100)`.
    - **Validates: Requirements 4.1**

- [x] 2. Implement `MatchingFlowTKG` constructor and context encoder
  - Define `MatchingFlowTKG(nn.Module)` in the same file.
  - `__init__(self, config, ode_steps=10)`:
    - Store `n_ent`, `n_rel`, `d_model`, `dropout_rate`, `ode_steps` from config.
    - Instantiate `self.encoder = GraphEmbedding(n_ent, n_rel, d_model, dropout, dropout, dropout)`.
    - Instantiate `self.vector_field = VectorField(d_model, dropout)`.
    - Instantiate `self.emb_regularizer = N3(0.004)`.
    - Add `self.w = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_model)).float(), requires_grad=True)`.
    - Add `self.time_mlp = nn.Linear(d_model, d_model)`.
    - Add `self.lp_loss_fn = nn.CrossEntropyLoss()`.
    - Add `self.dropout = nn.Dropout(dropout_rate)`.
  - Implement the internal `_encode_context(self, heads, rels, year, month, day)` helper:
    - Compute `time_signal = month + day % month + year % month`.
    - Compute `d_real = torch.sin(self.w.view(1,-1) * time_signal.unsqueeze(1))` and `d_img = torch.cos(...)` (kept for symmetric design parity; use `d_real` as the temporal input to CNN).
    - Retrieve `head_emb = self.encoder.get_ent_embedding(heads)` and `rel_emb = self.encoder.get_rel_embedding(rels)`.
    - Form `query_emb = head_emb + rel_emb`.
    - Call `context_emb = self.encoder(query_emb, d_real)` → `(bs, d_model)`.
    - Return `context_emb, head_emb, rel_emb`.
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 3.3_

  - [ ]* 2.1 Write property test for MatchingFlowTKG instantiation and signatures
    - **Property 1: MatchingFlowTKG instantiates for any valid config and produces correct method signatures**
    - For arbitrary (`n_ent`, `n_rel`, `d_model`, `dropout`) combinations, assert instantiation succeeds and both `train_forward` and `test_forward` are callable.
    - **Validates: Requirements 1.2, 1.3**

  - [ ]* 2.2 Write property test for context embedding dimension invariant
    - **Property 2: Context embedding dimension invariant**
    - For any batch of head/rel/temporal inputs and arbitrary `d_model`, assert `_encode_context(...)` returns a tensor of shape `(batch_size, d_model)`.
    - **Validates: Requirements 2.2, 3.1, 3.2, 10.3**

- [x] 3. Implement `train_forward`
  - Implement `train_forward(self, heads, rels, tails, year, month, day, neg)`:
    - Call `_encode_context` to get `context_emb`, `head_emb`, `rel_emb`.
    - Retrieve `x1 = self.encoder.get_ent_embedding(tails)` → `(bs, d_model)`.
    - Sample `t_scalar = torch.rand(bs, device=..., dtype=torch.float32)` → `(bs,)`.
    - Sample `noise = torch.randn_like(x1)` → `(bs, d_model)`.
    - Compute `x_t = (1 - t_scalar.unsqueeze(1)) * noise + t_scalar.unsqueeze(1) * x1`.
    - Compute target velocity `target_v = x1 - noise`.
    - Compute scalar flow-time embedding using sinusoidal frequencies (matching `NoName` scheme) and project through `self.time_mlp` → `t_emb (bs, d_model)`.
    - Compute `v_pred = self.vector_field(x_t, t_emb, context_emb)`.
    - Compute `fm_loss = F.mse_loss(v_pred, target_v)`.
    - Retrieve `neg_emb = self.encoder.get_ent_embedding(neg)` → `(bs, 500, d_model)`.
    - Compute link prediction scores: dot-product of `context_emb` against `x1` (positive) and `neg_emb` (negatives) → shape `(bs, 501)`.
    - Compute `lp_loss = self.lp_loss_fn(lp_scores, torch.zeros(bs, dtype=torch.long))`.
    - Compute N3 regularization factors from `head_emb`, `rel_emb`, `x1`: `reg_loss = self.emb_regularizer((head_emb, rel_emb, x1))`.
    - Return `fm_loss + lp_loss + reg_loss`.
  - _Requirements: 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 3.1 Write property test for flow-matching loss non-negativity and zero-at-identity
    - **Property 4: Flow-matching loss non-negativity and zero-at-identity**
    - For arbitrary training batches, assert `train_forward(...)` returns a non-negative scalar. Additionally construct a synthetic scenario where `v_pred == target_v` and assert the FM component equals zero.
    - **Validates: Requirements 4.4, 5.1, 5.2, 5.4, 5.5**

- [x] 4. Checkpoint — verify training path
  - Ensure `MatchingFlowTKG` can be instantiated, `train_forward` runs for a small synthetic batch without error, and the returned loss is a finite non-negative scalar.
  - Ask the user if any questions arise before proceeding to inference.

- [x] 5. Implement `test_forward` with Euler ODE solver
  - Implement `test_forward(self, heads, rels, tails, year, month, day)`:
    - Guard: if `self.ode_steps < 1`, raise `ValueError("ode_steps must be >= 1")`.
    - Compute `context_emb` via `_encode_context`.
    - Initialize `x_t = torch.randn(bs, d_model, device=...)`.
    - Run `ode_steps` Euler integration steps from `t=0` to `t=1` with step size `dt = 1.0 / ode_steps`:
      - For step `i`, compute `t_scalar = torch.full((bs,), i * dt, ...)`.
      - Embed `t_scalar` sinusoidally and project through `self.time_mlp`.
      - Compute velocity `v = self.vector_field(x_t, t_emb, context_emb)`.
      - Update `x_t = x_t + dt * v`.
    - Retrieve `all_ent_embs = self.encoder.get_all_ent_embedding()` → `(n_ent, d_model)`.
    - Compute scores as negative squared L2 distance: `scores = -torch.cdist(x_t, all_ent_embs).pow(2)` → `(bs, n_ent)`.
    - Return `scores`.
    - Ensure the method operates under `torch.no_grad()` when called from evaluation (no hard-coded `torch.no_grad()` wrapper; the caller in `main.py` already uses it).
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 10.2_

  - [ ]* 5.1 Write property test for test_forward output shape invariant
    - **Property 5: test_forward output shape invariant**
    - For arbitrary batch sizes and entity counts, assert `test_forward(...)` returns a tensor of shape `(batch_size, n_ent)`.
    - **Validates: Requirements 2.3, 6.1, 6.2, 8.1, 8.3, 10.2**

  - [ ]* 5.2 Write property test for highest-scoring entity is nearest neighbor
    - **Property 6: Highest-scoring entity is nearest neighbor**
    - For any test batch, assert that `test_forward(...).argmax(dim=1)` for each query equals the index of the entity embedding with the smallest L2 distance to the Euler-integrated embedding.
    - **Validates: Requirements 6.3**

- [x] 6. Update `main.py` with model registry and selection logic
  - Add `from models.MatchingFlow import MatchingFlowTKG` import at the top of `main.py`.
  - Define `MODEL_REGISTRY = {'DifTKG': NoName, 'MatchingFlow': MatchingFlowTKG}` immediately after imports.
  - In `main()`, replace the hard-coded `model = NoName(config)` line with:
    ```python
    if args.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model_name}'. Valid: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[args.model_name](config)
    ```
  - No other changes to `main.py`.
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 9.1_

  - [ ]* 6.1 Write property test for model registry dispatch and rejection
    - **Property 7: Model registry dispatches correctly and rejects unknown names**
    - For each key in `MODEL_REGISTRY`, assert the instantiated class matches the expected type. For arbitrary strings not in the registry, assert `ValueError` is raised.
    - **Validates: Requirements 7.1, 7.2, 7.3**

- [x] 7. Write unit tests for structural and boundary conditions
  - Create `tests/test_matching_flow_tkg.py` (or equivalent test file).
  - Include unit tests for:
    - `MatchingFlowTKG` is importable from `models.MatchingFlow`.
    - `model.encoder` is an instance of `GraphEmbedding`.
    - `model.w` is an `nn.Parameter` with `requires_grad=True`.
    - Flow interpolation correctness: at `t=0`, `x_t == noise`; at `t=1`, `x_t == x1` (within floating point tolerance).
    - `ode_steps=0` raises `ValueError`.
    - Unknown `--model_name` raises `ValueError` listing valid names.
  - _Requirements: 1.1, 1.3, 1.4, 3.3, 4.2, 4.3, 7.3_

- [x] 8. Final checkpoint — Ensure all tests pass
  - Ensure all unit tests and property tests pass.
  - Verify `python main.py --data icews14 --model_name MatchingFlow --do_train --max_epochs 1` runs without error (can be checked with a dry-run or single-batch sanity check in tests).
  - Verify `--model_name DifTKG` still works identically (no behavioral change to `NoName`).
  - Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP.
- Each task references specific requirements for traceability.
- `NoName` in `model.py`, `Denoiser.py`, `GraphEmbedding.py`, and `dataset.py` are never modified.
- Property tests use the `hypothesis` library with `@settings(max_examples=100)` and CPU-only tensors.
- The Euler step loop in `test_forward` is kept fully vectorized per step (no Python-level loops over entities).
