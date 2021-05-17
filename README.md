# Privacy Protected Machine Learning Optimization

This repository is the official implementation of Collaborative Learning with Privacy Protection, Provable Accuracy, and  Communication Efficiency.

## Requirements

To install requirements via anaconda utilize:

```setup
conda env create -f environment.yml
```
> The requirements.txt file is also avaliable
## Training
> Note that the model training is how the algorithm is evaluated through validation testing as iterations increase. Additionally, results were obtained from the average of 100 training runs for mnist and 25 runs for cifar10.

To train the base models for comparison, run these commands:

```train
python Train.py --data mnist --type base
python Train.py --data cifar10 --type base
```
> (train the model for the mnist and cifar10 datasets, respectively)



To train the privacy protected models, run these commands:
```train
python Train.py --data mnist --type new
python Train.py --data cifar10 --type new
```
>(train the model for the mnist and cifar10 datasets, respectively)

## Evaluation
> Before running the commands make two folders called "mnist" and "cifar10".

To evaluate the privacy algorithm against DLG attacks, run these commands:

```eval
python DLG.py --data mnist --type new
python DLG.py --data cifar10 --type new
```

To obtain comparison results for the non-privacy base algorithm against DLG attacks, run these commands:
```eval
python DLG.py --data mnist --type base
python DLG.py --data cifar10 --type base
```

## Pre-trained Models

You can download pretrained models here:



## Results

Our model achieves the following performance on :


| Algorithm name         | Top 1 Accuracy (mnist, CNN) | Top 1 Accuracy (cifar-10, ResNet-20)|
| ------------------ |---------------- | -------------- |
|Base Algorithm|     94.36%         |      72.16%       |
|Privacy Protection Algorithm|     98.66%         |      76.51%       |

