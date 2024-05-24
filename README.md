# Enhancing Adversarial Robustness in SNNs with Sparse Gradients 
**Yujia Liu, Tong Bu, Jianhao Ding, Zhaofei Yu, Tiejun Huang**

Implementation of Paper Enhancing Adversarial Robustness in SNNs with Sparse Gradients
(ICML 2024)

## Quick Start
### Requirements
Please first intall [torchattacks]([torchattacks · PyPI](https://pypi.org/project/torchattacks/)) before running this project.

### How to train
```bash
python snn_main_train --data_dir=[YOUR_DATA_PATH] --dataset=[cifar10|cifar100] --lamb=[Sparsity_Coefficient_Parameter] --model=[vgg11|wrn16]
```

### How to evaluate
You can add your own configuration file to perform multiple attacks in one run, or just perform one attack without specify --config
```bash
python snn_main_test --data_dir=[YOUR_DATA_PATH] --dataset=[cifar10|cifar100] --identifier=[FILENAME_TOBE_EVALUATED] --model=[vgg11|wrn16] --config=[JSON FILE NAME]
```

## Citations
If you find the code useful, please cite our work.

[TODO]