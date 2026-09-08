# NoName

Temporal knowledge graph (TKG) link prediction. A query is a quadruple `(head, relation, ?, timestamp)` and the model ranks candidate tail entities.

Two models share the same data pipeline and evaluation loop in `main.py`:

| `--model_name` | Class | Idea |
|---|---|---|
| `DifTKG` (default) | `NoName` in `model.py` | Discrete diffusion over entity embeddings |
| `MatchingFlow` | `MatchingFlowTKG` in `models/MatchingFlow.py` | Conditional flow matching (OT-CFM) with a Fourier context encoder |

## Setup

Python 3.8+ with PyTorch. GPU is used when CUDA is available.

```bash
pip install torch numpy tqdm pytest hypothesis
```

Place each dataset under `datasets/<name>/` as tab-separated `train.txt`, `valid.txt`, and `test.txt`:

```
head<TAB>relation<TAB>tail<TAB>YYYY-MM-DD
```

`datasets/icews14/` is included (ICEWS events from 2014). Other splits such as ICEWS05-15 and GDELT are described in [`datasets/README.md`](datasets/README.md).

`Dataset` always reads from `datasets/<name>/`. The `--data_root` flag is currently unused.

## Train and evaluate

```bash
# Diffusion baseline
python main.py --model_name DifTKG --data icews14 --do_train --do_test

# Matching flow (1-step Euler; fastest)
python main.py --model_name MatchingFlow --data icews14 --do_train --do_test

# Matching flow with extra Euler steps (slower, often higher MRR)
python main.py --model_name MatchingFlow --data icews14 --ode_steps 4 --do_train --do_test
```

Checkpoints are written to `output/<data>_<model_name>/model_<epoch>.pth` every `--valid_epoch` epochs (default 3). Logs go to `output/<data>_<model_name>/log.txt`.

Resume from a checkpoint:

```bash
python main.py --model_name MatchingFlow --data icews14 --do_test \
  --load_model_path output/icews14_MatchingFlow/model_30.pth
```

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `icews14` | Dataset folder name under `datasets/` |
| `--model_name` | `DifTKG` | `DifTKG` or `MatchingFlow` |
| `--d_model` | `200` | Embedding dimension |
| `--batch_size` | `256` | Train / eval batch size |
| `--max_epochs` | `31` | Training epochs |
| `--lr` | `0.001` | Adam learning rate |
| `--dropout` | `0.1` | Dropout |
| `--ode_steps` | `1` | MatchingFlow Euler steps (`1` is fastest; `4`–`8` can raise MRR) |
| `--valid_epoch` | `3` | Checkpoint + eval period during training |
| `--load_model_path` | `output1` | Path to a `.pth` file to restore |

## MatchingFlow

`MatchingFlowTKG` maps Gaussian noise to a tail embedding, conditioned on `(head, relation, year, month, day)`.

- Dual complex branches (`GraphEmbedding` tables + a Fourier transformer with FiLM time conditioning).
- Training loss: cross-entropy + contrastive (×5) + flow matching (×1.0) + N3 (×5) + frequency smoothness (×0.01).
- Default `--ode_steps 1` trains the vector field at `t = 0`, so the ranking reconstruction `x_hat = noise + v` matches one-step Euler at test. That is the fast path.
- `--ode_steps > 1` trains velocity at random `t` and integrates that many Euler steps at test.

The CNN inside `GraphEmbedding` is unused here; only the embedding tables are read.

## Evaluation

`test_forward` returns a score for every entity. `main.py` reports:

- **Raw** MRR and Hits@1 / 3 / 10
- **Static Filter** — other known true tails for the same `(head, relation)` are removed
- **Time Filter** — other true tails at the same timestamp are removed

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_matching_flow_tkg.py` covers construction, interpolation endpoints, and the `ode_steps=0` guard. `tests/test_vector_field_pbt.py` is a Hypothesis property test on `VectorField` output shapes.

## Layout

```
main.py                 train / eval entrypoint
model.py                DifTKG (NoName) diffusion model
models/MatchingFlow.py  MatchingFlowTKG
GraphEmbedding.py       entity / relation tables (+ unused CNN)
Denoiser.py             diffusion denoiser used by DifTKG
dataset.py              TKG loader and negative sampling
datasets/icews14/       ICEWS14 quadruples
regularizers.py         N3 cubic regularizer
tests/                  unit and property tests
```

`models/` also contains TransE, DistMult, Tero, and DE variants that are not wired into `MODEL_REGISTRY`.
