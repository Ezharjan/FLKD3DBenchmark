# Craniosynostosis

3D head surface scans from a statistical shape model of craniosynostosis patients — **100 mesh instances per pathology**, covering three suture-fusion types plus a control group.

> This folder ships empty (only a `.gitkeep`). The data is not redistributed here for licensing and size reasons — please download it from the official source below.

## Download & prepare

1. Open the Zenodo record: **https://zenodo.org/records/10167123**
2. Download the instance archive(s) (the `.ply` meshes; `.h5` shape models are optional).
3. Unzip and place the meshes into the matching `instances_*` folders shown below.

## Expected layout

```
craniosynostosis/
├── instances_control/    # normocephaly / positional plagiocephaly (healthy controls)
├── instances_coronal/    # coronal suture fusion  (brachy- / plagiocephaly)
├── instances_metopic/    # metopic suture fusion  (trigonocephaly)
├── instances_sagittal/   # sagittal suture fusion (scaphocephaly)
└── instances_full/       # all classes combined
```

Each `instances_*` folder holds **100 `.ply`** mesh instances.

## Citation

```bibtex
@article{schaufelberger2021statistical,
  title={A statistical shape model of craniosynostosis patients and 100 model instances of each pathology},
  author={Schaufelberger, Matthias and K{\"u}hle, RP and Wachter, A and Weichel, F and Hagen, N and Ringwald, F and Eisenmann, U and Hoffmann, J and Engel, M and Freudlsperger, C and others},
  journal={Zenodo: Geneve, Switzerland},
  year={2021}
}
```
