## Diachronic Embedding for Temporal Knowledge Graph Completion
This repository contains code for the reprsentation proposed in [Diachronic Embedding for Temporal Knowledge Graph Completion](https://arxiv.org/pdf/1907.03143.pdf) paper.
## Installation
- Create a conda environment:
```
$ conda create -n tkgc python=3.6 anaconda
```
- Run
```
$ source activate tkgc
```
- Change directory to TKGC folder
- Run
```
$ pip install -r requirements.txt
```
## How to use?
After installing the requirements, run the following command to reproduce results for DE-SimplE:
```
$ python main.py -dropout 0.4 -se_prop 0.68 -model DE-SimplE
```
To reproduce the results for DE-DistMult and DE-TransE, specify **model** as DE-DistMult/DE-TransE as following.
```
$ python main.py -dropout 0.4 -se_prop 0.36 -model DE_DistMult
$ python main.py -dropout 0.4 -se_prop 0.36 -model DE_TransE
```
MatchingFlow reuses the same dataset, trainer, and filtered ranking protocol:
```
$ python main.py -model MatchingFlow -dropout 0.1 -se_prop 1.0
$ python main.py -model MatchingFlow -dropout 0.1 -se_prop 1.0 -ode_steps 4
```
`-ode_steps 1` (default) is one-step OT-CFM. Use 4–8 for extra Euler refinement at test. `se_prop` only splits `emb_dim` into static/temporal sizes for DE models; MatchingFlow uses the full `emb_dim` as `d_model`.
## Citation
If you use the codes, please cite the following paper:
```
@inproceedings{goel2020diachronic,
  title={Diachronic Embedding for Temporal Knowledge Graph Completion},
  author={Goel, Rishab and Kazemi, Seyed Mehran and Brubaker, Marcus and Poupart, Pascal},
  booktitle={AAAI},
  year={2020}
}
```
## License
Copyright (c) 2018-present, Royal Bank of Canada.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
