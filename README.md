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

# Checkpoint
The model weights can be obtained here：[Checkpoint](https://drive.google.com/drive/folders/1GxG2A2lk3kxuv6fjldZ-5TNr5IM-ufZh)

# Training scripts
+ Train CIFAR100
```python train.py -seed 10 -epochs_base 200 -schedule 'Cosine' -lr_base 0.1 -dataset  'cifar100' -drop_rate 0.3 -std 0.01 -batch_size_base 256```

+ Train CUB200
 ```python train.py -seed 10 -epochs_base 200 -schedule 'Step' -lr_base 0.01 -dataset  'cub200' -drop_rate 0.8 -std 0.01 -batch_size_base 256```

+ Train miniImageNet
 ```python train.py -seed 10 -epochs_base 200 -schedule 'Cosine' -lr_base 0.1 -dataset  'mini_imagenet' -drop_rate 0.3 -std 0.01 -batch_size_base 256 ```

# Acknowledgment
We thank the following repos providing helpful components/functions in our work.
 + [CEC](https://github.com/icoz69/CEC-CVPR2021)
 + [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact?tab=readme-ov-file)

# Contact
If there are any questions, please feel free to contact with the author: [Kai Hu](kaihu@xtu.edu.cn). Enjoy the code.


