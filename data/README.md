# Datasets

This folder holds the five 3D datasets used in our experiments. To keep the repository light and respect each dataset's license, **the data itself is not included** — every folder ships empty (with a `.gitkeep`) and a short README telling you exactly where to download it and how to lay it out.

## How to use this folder

Pick a dataset below, open its folder, and follow the README. Each one gives you a download link, a copy-paste-ready layout, and the citation to use.

| Dataset | What it is | Contents | Get it |
|---|---|---|---|
| [`craniosynostosis/`](craniosynostosis/README.md) | Statistical-shape-model head scans of craniosynostosis patients | 4 classes × 100 mesh instances | [Zenodo](https://zenodo.org/records/10167123) |
| [`gazebosim/`](gazebosim/README.md) | Google Scanned Objects — real household items | SDF model packages | [Gazebo Fuel](https://app.gazebosim.org/dashboard) |
| [`modelnet40_normal_resampled/`](modelnet40_normal_resampled/README.md) | ModelNet40 point clouds (10k pts + normals) | 40 categories | [Kaggle](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled) |
| [`omni_object3d/`](omni_object3d/README.md) | OmniObject3D real-scanned objects | 216 categories | [Project page](https://omniobject3d.github.io/) |
| [`ycb/`](ycb/README.md) | YCB Object and Model Set for manipulation | Per-object meshes & clouds | [YCB benchmarks](https://ycb-benchmarks.s3.amazonaws.com/index.html) |

## Tips

- Keep each dataset in its existing folder name above — our code expects these paths.
- Datasets are large; download only the ones you need.
- Please cite the original authors (BibTeX is in each folder's README).
