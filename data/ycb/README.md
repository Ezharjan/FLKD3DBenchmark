# YCB Object and Model Set

Everyday objects scanned for robotic manipulation and perception benchmarking — meshes, textures, and point clouds from the **YCB Object and Model Set**.

> This folder ships empty (only a `.gitkeep`). The models are not redistributed here — please download them from the official source below.

## Download & prepare

1. Open the YCB downloads index: **https://ycb-benchmarks.s3.amazonaws.com/index.html**
2. Download the per-object archives you need (e.g. `google_16k` meshes and/or point clouds).
3. Extract each object into its own folder under `ycb/`.

## Expected layout

```
ycb/
├── <object_name>/         # e.g. 002_master_chef_can
│   ├── google_16k/        # textured mesh (.obj, .mtl, texture)
│   ├── clouds/            # point clouds
│   └── ...                # other scan products, if downloaded
└── ...                     # one folder per object
```

## Citation

```bibtex
@inproceedings{calli2015ycb,
  title={The ycb object and model set: Towards common benchmarks for manipulation research},
  author={Calli, Berk and Singh, Arjun and Walsman, Aaron and Srinivasa, Siddhartha and Abbeel, Pieter and Dollar, Aaron M},
  booktitle={2015 international conference on advanced robotics (ICAR)},
  pages={510--517},
  year={2015},
  organization={IEEE}
}
```
