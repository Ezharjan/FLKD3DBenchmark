# Gazebo Sim — Google Scanned Objects

High-quality 3D scans of real household items from **Google Scanned Objects (GSO)**, hosted on the Gazebo Fuel model library. Each model is an SDF package with meshes and textures.

> This folder ships empty (only a `.gitkeep`). The models are not redistributed here — please download them from Gazebo Fuel below.

## Download & prepare

1. Open Gazebo Fuel: **https://app.gazebosim.org/dashboard** (models live under the *GoogleResearch* collection).
2. Download the models you need (individually, or via the Fuel collection / `gz fuel download`).
3. Extract each model into its own folder under `gazebosim/`, keeping the SDF structure intact.

## Expected layout

```
gazebosim/
├── <object_name>/
│   ├── meshes/        # .obj / .dae geometry
│   ├── materials/     # textures
│   ├── model.config
│   └── model.sdf
└── ...                # one folder per object
```

## Citation

```bibtex
@inproceedings{downs2022google,
  title={Google scanned objects: A high-quality dataset of 3d scanned household items},
  author={Downs, Laura and Francis, Anthony and Koenig, Nate and Kinman, Brandon and Hickman, Ryan and Reymann, Krista and McHugh, Thomas B and Vanhoucke, Vincent},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)},
  pages={2553--2560},
  year={2022},
  organization={Ieee}
}
```
