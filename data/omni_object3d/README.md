# OmniObject3D

A subset of **OmniObject3D**, a large-vocabulary dataset of real-scanned 3D objects with high-fidelity geometry and texture. This folder is organized into **216 object categories** (`anise` … `zongzi`).

> This folder ships empty (only a `.gitkeep`). The data is not redistributed here — please download it from the official project page below.

## Download & prepare

1. Open the project page: **https://omniobject3d.github.io/** and follow its download instructions (data is served via OpenDataLab / OpenXLab).
2. Download the per-object scans / point clouds you need.
3. Extract them so each category sits in its own folder under `omni_object3d/`.

## Expected layout

```
omni_object3d/
├── anise/
├── antique/
├── apple/
├── ...                # 216 category folders in total
└── zongzi/
```

Each category folder holds that object's scans (per-instance point clouds / meshes).

## Citation

```bibtex
@inproceedings{wu2023omniobject3d,
  title={Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation},
  author={Wu, Tong and Zhang, Jiarui and Fu, Xiao and Wang, Yuxin and Ren, Jiawei and Pan, Liang and Wu, Wayne and Yang, Lei and Wang, Jiaqi and Qian, Chen and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={803--814},
  year={2023}
}
```
