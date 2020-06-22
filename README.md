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
- [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric)
- numpy
- [Eureqa](https://www.nutonian.com/download/eureqa-desktop-download/) (symbolic regression)

For simulations:

- [jax](https://github.com/google/jax) (simple N-body simulations)
- [quijote](https://github.com/franciscovillaescusa/Quijote-simulations) (Dark matter data)
- tqdm
- matplotlib

## Training

To train an example model from the paper, try out the [demo](https://drive.google.com/file/d/1mWK5LiSPYXG1qQBxvhwdLgvfhTrFBbsr/view?usp=sharing): 

> TODO 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> TODO 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

On the Newtonian simulations, each of these models achieve the following performance:

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
