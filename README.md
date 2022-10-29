# Robust Few-shot Learning Without Using any Adversarial Samples <br> (Under Review) - Official Implementation


### Dependencies:

    1. torch 1.10.2
    2. torchvision 0.11.3
    3. torchattacks 3.2.4
    4. tqdm 4.63.0
    5. fastai 1.0.58

<hr>

### Dataset Preparation

1. Download the dataset from [here](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view) and unzip it in ./CIFAR-FS/
2. Prepare the dataset using ./CIFAR-FS/prepare.py

<hr>

### Running Experiments

Step-1. Pretraining Stage

    1. Train the teacher model using ./scripts/pretrain_teacher.sh
    2. Train the student model using ./scripts/pretrain_student.sh

Step-2. Finetuning & Evaluation Stage

    1. Finetune and evaluate the pretrained student model using ./scripts/finetune.sh


<hr>

### Acknowledgements

This repo is adapted from [Dhillon et al. 2020](https://github.com/amazon-science/few-shot-baseline) and [Wang et al. 2020](https://github.com/HaohanWang/HFC)