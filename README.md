# A Python-Based Workflow for Asbestos Roof Mapping and Temporal Monitoring Using Satellite Imagery

This repository contains the source code accompanying the paper:

[**A Python-Based Workflow for Asbestos Roof Mapping and Temporal Monitoring Using Satellite Imagery**](https://doi.org/10.3390/geomatics6030041)

## Overview

This project provides a Python-based workflow for the detection, mapping, and temporal monitoring of asbestos roofs using satellite imagery. The repository includes the scripts required for data preprocessing, model training, inference, and result generation. The Preprocessing step indicated in the original work goes from notebook 0 to 4. The other steps correspond to their homonimous notebooks.

For clarity and ease of understanding, the code used in the paper is organized into Jupyter notebooks, which can be found in the **Notebooks** directory.

## Repository Structure

```text
.
├── Documentation/          # PDF copy of paper and WorldView-3 technical information.
├── Notebooks/              # Jupyter notebooks containing the presented workflow 
│   └── functions/          # Auxiliary scripts 
├── requirements.txt
└── README.md
```

## Requirements

The code was developed using **Python 3**.

The required Python packages are listed below and are also provided in the `requirements.txt` file included in this repository.

- [Py6S](https://py6s.readthedocs.io/en/latest/). *(we recommend installing it with conda, as this also installs all the required dependencies)*
- matplotlib
- geopandas
- pandas
- rasterio
- scikit-learn
- scikit-image

## Usage

The workflow is organized as a series of Jupyter notebooks.

- **Preprocessing:** Notebooks **0–4**.
- **Training:**  notebook **5 "Classification.ipynb"**
- **Postprocessing:** notebook **6 "PostProcessing.ipynb"**
- **Evaluation:** notebook **7 "AccuracyAssessment.ipynb"**

The notebooks should be executed in this order to reproduce the workflow presented in the paper.

## Data

The dataset used in the paper is not publicly available.

## Citation

If you use this repository or find it useful in your research, please cite the following paper:

```bibtex
@Article{Bonifazi2026_Geomatics,
AUTHOR = {Bonifazi, Giuseppe and Aurigemma, Alice and Salas-Cáceres, José and Lorenzo-Navarro, Javier and Serranti, Silvia and Paglietti, Federica and Bellagamba, Sergio and Malinconico, Sergio},
TITLE = {A Python-Based Workflow for Asbestos Roof Mapping and Temporal Monitoring Using Satellite Imagery},
JOURNAL = {Geomatics},
VOLUME = {6},
YEAR = {2026},
NUMBER = {3},
ARTICLE-NUMBER = {41},
URL = {https://www.mdpi.com/2673-7418/6/3/41},
ISSN = {2673-7418},
DOI = {10.3390/geomatics6030041}
}
```

## Contact

For questions regarding the code or the associated work, please contact:

**Alice Aurigemma**  
alice.aurigemma@uniroma1.it
