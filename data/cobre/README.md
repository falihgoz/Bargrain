## ACPI

### This folder should contain:
- ``COBRE_scan_data.tgz``
- ``COBRE_phenotypic_data.csv``

### To get the data:
- Download the data from this link: [https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)
- Choose ``COBRE_scan_data.tgz`` & ``COBRE_phenotypic_data.csv``
- Place those files into this folder
- Run this command:

```
python preprocess_data.py cobre
```