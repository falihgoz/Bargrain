# Bargrain

This is code implementation for Bargrain (Balanced graph structure for brains)

# Requirements
- torch==1.13.1+cu116
- nilearn==0.10.0
- pandas==1.3.5
- numpy==1.23.4
- omegaconf==2.3.0
- scikit-learn==1.1.2

# Dataset

The details to download the datasets are available in the ``./data/`` folder. Once the datasets are downloaded, preprocess all datasets using the command:

```
python preprocess_data.py acpi cobre abide
```

# Quick Start

```
python main.py --dataset <dataset>
```

# Citation

```
@inproceedings{febrinanto2023balanced,
  title={Balanced Graph Structure Information for Brain Disease Detection},
  author={Febrinanto, Falih Gozi and Liu, Mujie and Xia, Feng},
  booktitle={Pacific Rim Knowledge Acquisition Workshop},
  pages={134--143},
  year={2023},
  organization={Springer}
}
```