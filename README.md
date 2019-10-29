# Multi-view 3D recognition by Mostafa and Tan
This repository holds the codes and data for multi-view 3D recognition

## Getting Started

### Prerequisites
Pytorch

```
conda install pytorch torchvision -c pytorch
```

### Data Prepare

You need download ModelNet40 dataset

with orientation assumption, 12-view settings
```
wget http://www.cim.mcgill.ca/dscnn-data/ModelNet40_rendered_rgb.tar; tar -xvf ModelNet40_rendered_rgb.tar 
```

without orientation assumption, 20-view settings
```
 wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar; tar -xvf modelnet40v2png_ori4.tar
```


## Training

12-view ModelNet40
```
python mainalex.py -d modelnet40  
```

12-view ModelNet10
```
python mainalex.py -d modelnet10 
```




