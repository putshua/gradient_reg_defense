# Enhancing Adversarial Robustness in SNNs with Sparse Gradients 
**Yujia Liu, Tong Bu, Jianhao Ding, Zhaofei Yu, Tiejun Huang**

Implementation of Paper Enhancing Adversarial Robustness in SNNs with Sparse Gradients
(ICML 2024)

## Quick Start
### Requirements
Please first intall [torchattacks](https://pypi.org/project/torchattacks/) before running this project (torchattacks 3.2.7).

### How to train
```bash
python snn_main_train --data_dir=[YOUR_DATA_PATH] --dataset=[cifar10|cifar100] --lamb=[Sparsity_Coefficient_Parameter] --model=[vgg11|wrn16]
```

### How to evaluate
You can add your own configuration file to perform multiple attacks in one run, or just perform one attack without specify --config
```bash
python snn_main_test --data_dir=[YOUR_DATA_PATH] --dataset=[cifar10|cifar100] --identifier=[FILENAME_TOBE_EVALUATED] --model=[vgg11|wrn16] --config=[JSON FILE NAME]
```

### checkpoints
Download from [checkpoints](https://drive.google.com/drive/folders/196nQUljY01lCV5LYlgx-f5MK366uIId2?usp=sharing)

More checkpoints in [site2](https://disk.pku.edu.cn/link/AABD1D2719359B49E29BF2AA70FB6F1448)

Move all checkpoints into directory `./[DATASETNAME]-checkpoints/` , e.g. `./cifar10-checkpoints/` before evaluating.

## Citations
If you find the code useful, please cite our work.

@inproceedings{liu2024enhancing,
  title={Enhancing Adversarial Robustness in SNNs with Sparse Gradients},
  author={Liu, Yujia and Bu, Tong and Ding, Jianhao and Hao, Zecheng and Huang, Tiejun and Yu, Zhaofei},
  booktitle={International Conference on Machine Learning},
  pages={30738--30754},
  year={2024},
  organization={PMLR}
}
