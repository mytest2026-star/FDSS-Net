<<<<<<< Updated upstream
# FDSS-Net：Feature Enhancement and Dual-Stream Semantic Mixture Network for Polyp Segmentation
The following information was supplied regarding data availability:
Kvasir-SEG is available at Simula:https://datasets.simula.no/kvasir-seg/.
CVC-ClinicDB is available at Simula:https://polyp.grand-challenge.org/CVCClinicDB/.
CVC-ColonDB is available at Simula:http://vi.cvc.uab.es/colon-qa/cvccolondb.
CVC-300 is available at Simula:https://www.kaggle.com/datasets/doansang/cvc-300.
ETIS-LaribPolypDB is available at Simula:https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb.

The command for the training dataset is "python train.py", and the command for the testing dataset is "python test.py".
=======
## FDSS-Net：Feature Enhancement and Dual-Stream Semantic Mixture Network for Polyp Segmentation

We proposed FEDM-Net architecture consists of the backbone PVT, Feature  Enhancement and Propagation Module (FEPM), Dual-Stream Semantic Mixture (DSSM) Module, and Hierarchical Multi-Scale Aggregation and Prediction (HMAP) module.

To quantitatively evaluate the segmentation performance and generalizability of our method, we employ five publicly available polyp segmentation datasets: ColonDB, ETIS-LaribPolypDB, Kvasir, CVC-300, and ClinicDB. The training set is constructed by combining 900 images from Kvasir and 550 images from ClinicDB, resulting in 1,450 labeled samples. All other datasets are exclusively used during testing to assess the model's generalization capability across domains, particularly under varying imaging conditions and distribution shifts.

The details of this project are presented in the following paper.


## Usage 
### Setup 
```
Python 3.12
Pytorch 2.8.0
torchvision 0.23
```
### Dataset 
Download the training and test datasets and move them into `./dataset/`, see [Google Drive](https://drive.google.com/file/d/16o4vcTvclsddlqBXMK80j8rT99qK3Hr3/view?usp=drive_link).

### Pre-trained model 
Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1Fdz23p1NW0jy3JUhx1_Lqf4p8JEbp4io/view?usp=drive_link), and then put it in the `./pretrained_pth`  folder for initialization. 

### Train the model 
Clone the repository
```
git clone https://github.com/mytest2026-star/FDSS-Net.git
cd FDSSNet 
python train.py
```

### Test the model
```
cd FDSSNet 
python test.py
```

### Evaluate the trained model 

```
cd FDSSNet 
python eval.py
```


## Acknowledgement
Thanks [HSNet](https://github.com/baiboat/HSNet), [CAFENet](https://github.com/shenjoyao/CAFE-Net)and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) for serving as building blocks of FDSSNet.

## Citation

If you find our work/code interesting, welcome to cite our paper

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.
>>>>>>> Stashed changes
