# Docs: Detecting outlier cells and subtypes for single-cell transcriptomics via adversarial training
We proposed a GAN-based model named <b>Docs</b> (<b>D</b>etecting <b>o</b>utlier <b>c</b>ells and <b>s</b>ubtypes). This approach employs a pipeline to integrate multi-task generative adversarial networks for detecting outlier cells and the subtypes of these cells in single-cell transcriptomics (scRNA-seq and scATAC-seq data). 

## Download and unzip Datasets
- Download needed datasets from this link: [ODDatasets](https://drive.google.com/drive/folders/1-jHkZweZC0nJPUZcutzJqoRxL-Yvz57q?usp=drive_link).
- Unzip the ODDatasets.zip file.
- All the datasets are stored as 'h5ad', and read by [Scanpy](https://scanpy.readthedocs.io/en/stable/).

### Data organization

```
ODDatasets
|
└───PBMC
    |  PBMC_3000_ref.h5ad
    |  PBMC_3000_B.h5ad
    |  ...
|
└───Cancer
    |  Cancer_3000_ref.h5ad
    |  Cancer_3000_EI.h5ad
    |  ...

```

### scRNA-seq PBMC Dataset Information

|Dataset|Cells|Genes|Sparsity ratio(%)|Outlier ratio(%)|Outlier Type|
|:---:|:---:|:---:|:---:|:---:|:---:|
|PBMC_3000_ref|4698|3000|96.82|| |
|PBMC_3000_B|3684|3000|96.89|13.14|B cells|
|PBMC_3000_NK|3253|3000|96.48|12.73|NK cells|
|PBMC_6000_ref|4698|6000|97.52| | |
|PBMC_6000_B|3684|6000|97.55|13.14|B cells|
|PBMC_6000_NK|3253|6000|97.27|12.73|NK cells|
|PBMC_full_ref|4698|32738|98.82| | |
|PBMC_full_B|3684|32738|98.83|13.14|B cells|
|PBMC_full_NK|3253|32738|98.71|12.73|NK cells|

### scRNA-seq Lung Cancer Dataset Information

|Dataset|Cells|Genes|Sparsity ratio(%)|Outlier ratio(%)|Outlier Type|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Cancer_3000_ref|8104|3000|94.94| | |
|Cancer_3000_EI|7721|3000|96.08|50.03|Epithelial & Immune Tumor|
|Cancer_3000_ES|4950|3000|94.86|58.12|Epithelial & Stromal Tumor|
|Cancer_6000_ref|8104|6000|94.85| | |
|Cancer_6000_EI|7721|6000|95.99|50.03|Epithelial & Immune Tumor|
|Cancer_6000_ES|4950|6000|94.32|58.12|Epithelial & Stromal Tumor|
|Cancer_full_ref|8104|33538|92.90| | |
|Cancer_full_EI|7721|33538|94.68|50.03|Epithelial & Immune Tumor|
|Cancer_full_ES|4950|33538|93.15|58.12|Epithelial & Stromal Tumor|

### scATAC-seq TME Dataset Information

|Dataset|Cells|Genes|Sparsity ratio(%)|Outlier ratio(%)|Outlier Type|
|:---:|:---:|:---:|:---:|:---:|:---:|




