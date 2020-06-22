# [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/30000000000TODO)

This repository is the official implementation of [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/30000000000TODO). 

Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, Shirley Ho
- [Blog](https://astroautomata.com/paper/symbolic-neural-nets/)
- [Paper](https://arxiv.org/abs/30000000000TODO)
- [Video](https://youtu.be/2vwwu59RPL8)
- [Interactive Demo](https://arxiv.org/abs/30000000000TODO)
- [Repo](https://github.com/MilesCranmer/symbolic_deep_learning)

[![](images/discovering_symbolic_eqn_gn.png)](https://astroautomata.com/paper/symbolic-neural-nets/)


## Requirements

For model:

- pytorch
- pytorch-geometric
- numpy

For simulations:

- [jax](https://github.com/google/jax) (simple N-body simulations)
- [quijote](https://github.com/franciscovillaescusa/Quijote-simulations) (Dark matter data)
- tqdm
- matplotlib

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
