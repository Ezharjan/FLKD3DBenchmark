# ModelNet40 (Normal, Resampled)

ModelNet40 CAD models resampled to **10,000 points each with surface normals** (`x, y, z, nx, ny, nz`), across **40 object categories** — the preprocessed format used by PointNet++.

> This folder ships empty (only a `.gitkeep`). The data is not redistributed here for size reasons — please download it from the source below.

## Download & prepare

1. Open the Kaggle dataset: **https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled**
2. Download and unzip it.
3. Place the 40 category folders **and** the split `.txt` files directly under `modelnet40_normal_resampled/` (see below).

## Expected layout

```
modelnet40_normal_resampled/
├── airplane/
│   ├── airplane_0001.txt     # 10,000 rows: x,y,z,nx,ny,nz
│   └── ...
├── bathtub/
├── ...                        # 40 category folders in total (airplane … xbox)
├── modelnet40_shape_names.txt
├── modelnet40_train.txt
├── modelnet40_test.txt
└── filelist.txt
```

## Citation

```bibtex
% (Preprocessed dataset):
@article{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}


% (Original dataset):
@inproceedings{wu20153d,
  title={3d shapenets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1912--1920},
  year={2015}
}
```
