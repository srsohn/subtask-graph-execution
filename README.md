# Subtask-graph-maze

<p align="center">
<img src="doc/playground_slow.gif" title="*"/>
</p>

This repository is a python implementation of Playground and Mining domain in [NeurIPS 2018 Hierarchical Reinforcement Learning for Zero-shot Generalization with Subtask Dependencies](https://arxiv.org/pdf/1807.07665.pdf).
```
@inproceedings{sohn2018hierarchical,
  title={Hierarchical Reinforcement Learning for Zero-shot Generalization with Subtask Dependencies},
  author={Sohn, Sungryull and Oh, Junhyuk and Lee, Honglak},
  booktitle={Advances in Neural Information Processing Systems},
  pages={7156--7166},
  year={2018}
}
```
# Requirements
* Python 3 (it might work with Python 2, but I didn't test it)
* Pygame, graphviz (for interactive visualization only)


# Installation
You can perform a minimal installation of SGE with:
```
git clone https://github.com/srsohn/subtask-graph-maze.git
cd subtask-graph-maze
pip install .
```
If you want interactive demo with visualization:
```
pip install .[visualize]
```


# Interactive demo with visualization
The following command runs interactive demonstration of Playground environment with 'D3' graph set (see the paper):
```
python demo_visual.py --game_name playground --graph_param D3_eval_1
```

The following command runs interactive demonstration of Mining environment with 'train' graph set (see the paper):
```
python demo_visual.py --game_name mining --graph_param train_1
```

# Demo of random policy without visualization
The following command runs demonstration of random policy in Playground environment with 'D2' graph set (see the paper):
```
python demo_random.py --game_name playground --graph_param D2_eval_1
```

The following command runs demonstration of random policy in Mining environment with 'eval' graph set (see the paper):
```
python demo_random.py --game_name mining --graph_param eval_1
```

# Icons
The icons used in Mining domain were downloaded from www.flaticon.com.
