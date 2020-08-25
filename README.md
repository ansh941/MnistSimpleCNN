## MnistSimpleCNN

This repository is implementation of "An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition".

Paper url is https://arxiv.org/abs/2008.10400.

In paper, we propose simple models classifying MNIST called M3, M5, M7 following kernel size.

### Train

```bash
python3 train.py --seed=0 --trial=10 --kernel_size=5 --gpu=0 --logdir=modelM5
```

Parameters:

seed : random seed number

trial : the number of trial. When previous trial is end, add present trial number to seed number.

Ex) seed=0 trial=10 ⇒ execute seed 0~9

kernel_size : kernel size of model. You can select the model following this parameter.

gpu : gpu number. You can use only one gpu during training in this code, but can select gpu when you training.

logdir : save directory address name. It makes a sub-directory using that name at logs directory.

### Test

```bash
python3 test.py  --seed=0 --trial=10 --kernel_size=5 --logdir=modelM5
```

test.py loads model saving files and make wrong image number list for each seed.

### Ensemble

```bash
python3 homo_ensemble.py --kernel_size=5
```

homo_ensemble.py loads wrong image number list files of same model saving during executing test.py. And then calculate the accuracy of ensemble model through majority voting.
