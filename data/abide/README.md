## ABIDE

### This folder should contain:
- .\ABIDE_pcp
- Phenotypic_V1_0b_preprocessed1.csv

### To get the data:
- Run this command:
```
python fetch_data.py
```

- Once files are generated in .\data\abide, run this command:

```
python preprocess_data.py abide
```

### Referece:

```bibtex
@article{di2014autism,
  title={The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism},
  author={Di Martino, Adriana and Yan, Chao-Gan and Li, Qingyang and Denio, Erin and Castellanos, Francisco X and Alaerts, Kaat and Anderson, Jeffrey S and Assaf, Michal and Bookheimer, Susan Y and Dapretto, Mirella and others},
  journal={Molecular psychiatry},
  volume={19},
  number={6},
  pages={659--667},
  year={2014},
  publisher={Nature Publishing Group}
}
```
