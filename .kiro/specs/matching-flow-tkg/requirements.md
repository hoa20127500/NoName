# Requirements Document

## Introduction

This feature adds a **Matching Flow** model (`MatchingFlowTKG`) to the existing temporal knowledge graph (TKG) reasoning system. The current `NoName` model uses a discrete diffusion process for link prediction and serves as the benchmark. The new model uses a continuous normalizing flow / flow-matching approach to learn a mapping from a source distribution (Gaussian noise) to the target distribution over tail entity embeddings, conditioned on the query context (head entity, relation, and temporal encoding of year/month/day). The goal is to offer an alternative generative approach that can be selected at runtime and compared against the diffusion baseline using the same training loop, evaluation metrics, and dataset pipeline.

---

## Glossary

- **TKG**: Temporal Knowledge Graph — a knowledge graph where each fact (head, relation, tail) is associated with a timestamp (year, month, day).
- **Quadruple**: A single TKG fact of the form `(head, relation, tail, year, month, day)`.
- **MatchingFlowTKG**: The new flow-matching-based model class introduced by this feature.
- **NoName**: The existing diffusion-based TKG model that acts as the benchmark. Must not be modified.
- **Flow**: A continuous normalizing flow network that models a probability path between a source distribution and a target distribution.
- **VectorField**: The neural network inside MatchingFlowTKG that predicts the velocity field used to integrate the flow trajectory.
- **ContextEncoder**: The component (reusing `GraphEmbedding`) that produces a conditioning vector from (head, relation, time) inputs.
- **GraphEmbedding**: The existing CNN-based embedding module in `GraphEmbedding.py` providing entity and relation embeddings.
- **N3**: The existing cubic regularizer in `regularizers.py`.
- **LinkPredictionScore**: The scalar score assigned to each candidate tail entity during evaluation, used to rank all entities.
- **MRR**: Mean Reciprocal Rank — the primary evaluation metric.
- **HITS@K**: The fraction of queries where the correct tail is ranked within the top K entities.
- **Static Filter**: Evaluation protocol that removes all known true answers (across all timestamps) except the query answer before ranking.
- **Time Filter**: Evaluation protocol that removes all true answers at the exact query timestamp except the query answer before ranking.
- **train_forward**: The method called during training that returns a scalar loss.
- **test_forward**: The method called during evaluation that returns a score tensor of shape `(batch_size, n_ent)`.

---

## Requirements

### Requirement 1: MatchingFlowTKG Model Class

**User Story:** As a researcher, I want a `MatchingFlowTKG` class in `models/MatchingFlow.py` so that I can use a flow-matching approach for TKG link prediction without modifying the existing diffusion model.

#### Acceptance Criteria

1. THE `MatchingFlowTKG` SHALL be defined as a `torch.nn.Module` subclass in a new file `models/MatchingFlow.py`.
2. THE `MatchingFlowTKG` SHALL accept a `config` object with attributes `n_ent`, `n_rel`, `d_model`, and `dropout` — the same config interface used by `NoName`.
3. THE `MatchingFlowTKG` SHALL expose a `train_forward(heads, rels, tails, year, month, day, neg)` method and a `test_forward(heads, rels, tails, year, month, day)` method matching the signatures used in `main.py`.
4. THE `MatchingFlowTKG` SHALL NOT import from or modify `model.py`, `Denoiser.py`, or any existing baseline model file.

---

### Requirement 2: Reuse of GraphEmbedding

**User Story:** As a researcher, I want `MatchingFlowTKG` to reuse `GraphEmbedding` for entity and relation embeddings so that the embedding approach is consistent with the benchmark and the embedding space is shared.

#### Acceptance Criteria

1. THE `MatchingFlowTKG` SHALL instantiate at least one `GraphEmbedding` object (imported from `GraphEmbedding.py`) to provide entity and relation embeddings.
2. WHEN computing the context vector for a query `(head, relation, time)`, THE `ContextEncoder` SHALL use `GraphEmbedding.get_ent_embedding`, `GraphEmbedding.get_rel_embedding`, and `GraphEmbedding.forward` to produce a conditioning embedding of dimension `d_model`.
3. WHEN computing scores over all candidate tail entities during evaluation, THE `MatchingFlowTKG` SHALL use `GraphEmbedding.get_all_ent_embedding` to retrieve the full entity embedding matrix.

---

### Requirement 3: Temporal Context Encoding

**User Story:** As a researcher, I want the flow to be conditioned on the temporal context (year, month, day) so that the model captures time-varying relational patterns in the TKG.

#### Acceptance Criteria

1. THE `MatchingFlowTKG` SHALL encode the temporal information `(year, month, day)` into a time embedding vector of dimension `d_model` using sinusoidal or learnable frequency encoding before conditioning the flow.
2. WHEN constructing the context vector, THE `ContextEncoder` SHALL combine the temporal embedding with the head entity embedding and the relation embedding to produce a single conditioning vector of dimension `d_model`.
3. THE temporal encoding weights SHALL be learnable parameters of `MatchingFlowTKG` and SHALL be updated during training.

---

### Requirement 4: Flow-Matching Vector Field

**User Story:** As a researcher, I want the model to use a flow-matching (conditional flow matching) approach with a neural vector field so that the mapping from noise to tail embeddings is learned via a simple regression objective rather than a complex ELBO or score-matching loss.

#### Acceptance Criteria

1. THE `VectorField` SHALL be a neural network (MLP or attention-based) that takes as input the concatenation of `(noised_embedding, time_scalar_embedding, context_embedding)` and outputs a vector of dimension `d_model` representing the predicted velocity.
2. WHEN training, THE `MatchingFlowTKG` SHALL sample a scalar flow time `t` uniformly from `[0, 1]` for each sample in the batch.
3. WHEN training, THE `MatchingFlowTKG` SHALL construct the interpolated sample `x_t = (1 - t) * noise + t * x_1` where `noise` is sampled from a standard Gaussian and `x_1` is the target tail entity embedding.
4. WHEN training, THE `MatchingFlowTKG` SHALL compute the conditional flow matching loss as the mean squared error between the `VectorField` output and the target velocity `(x_1 - noise)`.
5. THE `VectorField` SHALL have at least two hidden layers with non-linear activations and SHALL support dropout regularization using the `dropout` value from `config`.

---

### Requirement 5: Training Loss

**User Story:** As a researcher, I want the training loss to combine the flow-matching objective with a link prediction loss so that the model is optimized both for density estimation and for ranking the correct tail entity.

#### Acceptance Criteria

1. WHEN `train_forward` is called, THE `MatchingFlowTKG` SHALL compute a flow-matching regression loss (MSE between predicted velocity and target velocity) over the positive tail embeddings.
2. WHEN `train_forward` is called, THE `MatchingFlowTKG` SHALL compute a link prediction loss (`CrossEntropyLoss`) over the scores of the positive tail and the `neg` negative samples provided as input.
3. WHEN `train_forward` is called, THE `MatchingFlowTKG` SHALL compute an N3 regularization term (using the existing `N3` class from `regularizers.py`) on the entity and relation embedding factors.
4. THE `train_forward` method SHALL return the scalar sum of the flow-matching loss, the link prediction loss, and the N3 regularization term.
5. WHEN `train_forward` is called, THE `MatchingFlowTKG` SHALL accept negative samples `neg` of shape `(batch_size, num_neg)` as provided by `QuadruplesDataset.__getitem__`, consistent with the `NoName` interface.

---

### Requirement 6: Test-Time Score Computation

**User Story:** As a researcher, I want `test_forward` to produce a score for every entity in the knowledge graph so that the standard ranking evaluation in `main.py` works without modification.

#### Acceptance Criteria

1. WHEN `test_forward` is called, THE `MatchingFlowTKG` SHALL generate a denoised entity embedding by integrating the `VectorField` from `t=0` to `t=1` using a fixed-step Euler ODE solver with a configurable number of integration steps (default: 10).
2. WHEN `test_forward` is called, THE `MatchingFlowTKG` SHALL return a score tensor of shape `(batch_size, n_ent)` where `n_ent` matches `config.n_ent`.
3. THE score for each candidate entity SHALL be computed as the negative L2 distance (or dot-product similarity) between the integrated embedding and each entity embedding from `GraphEmbedding.get_all_ent_embedding`.
4. WHEN `test_forward` is called, THE `MatchingFlowTKG` SHALL operate under `torch.no_grad()` to prevent unnecessary gradient computation during inference.

---

### Requirement 7: Integration into main.py

**User Story:** As a researcher, I want to select `MatchingFlowTKG` via the `--model_name` CLI argument so that I can run training and evaluation experiments without modifying source code.

#### Acceptance Criteria

1. WHEN `--model_name MatchingFlow` is passed, THE `main.py` SHALL instantiate `MatchingFlowTKG` instead of `NoName`.
2. WHEN `--model_name DifTKG` (the current default) is passed, THE `main.py` SHALL instantiate `NoName` exactly as it does today, with no behavioral change.
3. IF an unrecognized `--model_name` value is passed, THEN THE `main.py` SHALL raise a `ValueError` with a message listing the valid model names.
4. THE `main.py` training loop, evaluation loop, checkpoint saving, and metric logging SHALL require no structural changes beyond the model selection logic.

---

### Requirement 8: Evaluation Metrics Compatibility

**User Story:** As a researcher, I want `MatchingFlowTKG` to be evaluated with the same metrics (Raw MRR, HITS@1/3/10, Static Filter MRR/HITS, Time Filter MRR/HITS) as `NoName` so that results are directly comparable.

#### Acceptance Criteria

1. THE score tensor returned by `MatchingFlowTKG.test_forward` SHALL have shape `(batch_size, n_ent)` with one score per entity, compatible with the rank computation in `test()` in `main.py`.
2. WHEN scores are sorted in descending order in `test()`, THE correct tail entity's position SHALL determine its rank, consistent with how `NoName` is evaluated.
3. THE `MatchingFlowTKG` SHALL support both Static Filter and Time Filter evaluation as implemented in the existing `test()` function without any modification to that function.

---

### Requirement 9: Benchmark Preservation

**User Story:** As a researcher, I want the existing `NoName` diffusion model to remain completely unchanged so that I can use it as a reliable baseline for comparison.

#### Acceptance Criteria

1. THE `model.py` file SHALL NOT be modified by this feature.
2. THE `Denoiser.py` file SHALL NOT be modified by this feature.
3. THE `GraphEmbedding.py` file SHALL NOT be modified by this feature.
4. THE `dataset.py` file SHALL NOT be modified by this feature.
5. WHEN `--model_name DifTKG` is used, THE system SHALL produce identical training and evaluation behavior to the pre-feature state.

---

### Requirement 10: Round-Trip Embedding Consistency

**User Story:** As a researcher, I want the embedding encoding and decoding to be internally consistent so that the flow operates in a well-defined embedding space.

#### Acceptance Criteria

1. THE entity embeddings produced by `GraphEmbedding.get_ent_embedding` SHALL have dimension `d_model`, consistent with the flow's input and output dimension.
2. THE integrated embedding produced by `test_forward` SHALL have the same dimension `d_model` as the entity embeddings used for scoring, ensuring dimensional consistency throughout the pipeline.
3. FOR ALL valid input quadruples, the context embedding dimension used as flow conditioning SHALL equal `d_model`.
