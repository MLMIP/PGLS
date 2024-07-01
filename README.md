# Progressive Learning Strategy for Few-Shot Class-Incremental Learning (PGLS)

The code repository for "Progressive Learning Strategy for Few-Shot Class-Incremental Learning "
![image](https://github.com/MLMIP/PGLS/assets/67742308/392915fc-3598-4609-846e-263dd9d3422e)


# Results
![image](https://github.com/MLMIP/PGLS/assets/67742308/929b9c97-a644-4e84-bdd2-f897ac13489a)

Please refer to our paper for detailed values.

# Prerequisites
The following packages are required to run the scripts:
+ PyTorch-1.4 and torchvision
+ tqdm

# Dateset
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.

# Code Structures
There are four parts in the code.

+ ```models```:It contains the backbone network and training protocols for the experiment.
+ ```data```: Images and splits for the data sets.
+ ```dataloader```: Dataloader of different datasets.
+ ```checkpoint```:The weights and logs of the experiment.
