# [Path Integral Sampler: a stochastic control approach for sampling](https://arxiv.org/abs/2111.15141)


[![PIS Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NOHGt2iHoETgbSh4z7dLep95-zN8FNKL?usp=sharing)

![PIS](asserts/pis.png)

## Setup

The repo heavily depends on [jam](https://github.com/qsh-zh/jam), a versatile toolbox developed by [Qsh.zh](https://github.com/qsh-zh) and [jam-hlt](https://github.com/qsh-zh/jam), a decent deep leanring project template. [⭐️](https://github.com/qsh-zh/jam) if you like them.

*[poetry](https://python-poetry.org/)* (**Recommended**)
```shell
curl -fsS -o /tmp/get-poetry.py https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py
python3 /tmp/get-poetry.py -y --no-modify-path
export PATH=$HOME/.poetry/bin:$PATH
poetry shell
poetry install
```

*pip*
```shell
pip install .
```

## Reproduce

```
python run.py experiment=ou.yaml logger=wandb
```

See the [folder](configs/experiment) for more experiment configs.

There are some [results](https://wandb.ai/qinsheng/pub_pis?workspace=user-qinsheng) reproduced by the repo.

## Reference

```tex
@inproceedings{zhang2021path,
  author    = {Qinsheng Zhang and Yongxin Chen},
  title     = {Path Integral Sampler: a stochastic control approach for sampling},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```
