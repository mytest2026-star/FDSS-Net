<<<<<<< Updated upstream
# FDSS-Net：Feature Enhancement and Dual-Stream Semantic Mixture Network for Polyp Segmentation
To quantitatively evaluate the segmentation performance and generalizability of our method, we employ five publicly available polyp segmentation datasets: ColonDB, ETIS-LaribPolypDB, Kvasir, CVC-300, and ClinicDB. 
The training set is constructed by combining 900 images from Kvasir and 550 images from ClinicDB, resulting in 1,450 labeled samples. All other datasets are exclusively used during testing to assess the model's generalization capability across domains, particularly under varying imaging conditions and distribution shifts.

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
Download the training and test datasets and move them into `./dataset/`, see [FigShare](https://doi.org/10.6084/m9.figshare.31363015).

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
