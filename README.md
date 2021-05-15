# Privacy Protected Machine Learning Optimization

This repository is the official implementation of [Name of Paper](https://link.com). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements via anaconda utilize:

```setup
conda create --name <name of env> --file requirements.txt
```

## Training

To train the base models for comparison, run these commands:

```train
python Train.py mnist base
python Train.py cifar10 base
```
> (train the model for the mnist and cifar10 datasets, respectively)



To train the privacy protected models, run these commands:
(train the model for the mnist and cifar10 datasets, respectively)
```train
python Train.py mnist new
```
```train
python Train.py cifar10 new
```
>(train the model for the mnist and cifar10 datasets, respectively)

## Evaluation

To evaluate the algorithm against DLG attacks, run these commands:

```eval
python DLG.py --data mnist --type new
```
```eval
python DLG.py --data cifar10 --type new
```

To obtain comparison results for the non-privacy base algorithm, run these commands:
```eval
python DLG.py --data mnist --type base
```
```eval
python DLG.py --data cifar10 --type base
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

