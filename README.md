
# The Demo of [DKEPool](https://github.com/chenchkx/DKEPool) Applied in Transient Stability Assessment of Power System

This is the code for our paper "Distribution-Aware Graph Representation Learning for Transient Stability Assessment of Power System, IJCNN2022". It is based on the code from [SOPool](https://github.com/divelab/sopool). Many thanks!

Created by [Kaixuan Chen](chenkx@zju.edu.cn) (chenkx@zju.edu.cn, chenkx.jsh@aliyun.com)

## Download & Citation

If you find our code useful for your research, please kindly cite our paper.
```
@inproceedings{chen2022distribution,
  title={Distribution-Aware Graph Representation Learning for Transient Stability Assessment of Power System},
  author={Chen, Kaixuan and Liu, Shunyu and Yu, Na and Yan, Rong and Zhang, Quan and Song, Jie and Feng, Zunlei and Song, Mingli},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  year={2022},
  organization={IEEE}
}

@article{chen2022DKEPool,
  title={Distribution Knowledge Embedding for Graph Pooling},
  author={Chen, Kaixuan and Song, Jie and Liu, Shunyu and Yu, Na and Feng, Zunlei and Han, Gengshi and Song, Mingli},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022}
}
```

## System requirement

#### Programming language

```
Python 3.6
```

#### Python Packages

```
PyTorch > 1.0.0, tqdm, networkx, numpy
```

## Run the code

We provide scripts to run the experiments.

For DKEPool module tested on IEEE 39-BUS dataset, run

```
sh sh_case39.sh
```
