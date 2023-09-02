# Docs: Detecting outlier cells and subtypes for single-cell transcriptomics via adversarial training.
We proposed a GAN-based model named <b>Docs</b> (<b>D</b>etecting <b>o</b>utlier <b>c</b>ells and <b>s</b>ubtypes). This approach employs a pipeline to integrate multi-task generative adversarial networks for detecting outlier cells and the subtypes of these cells in single-cell transcriptomics (scRNA-seq and scATAC-seq data). 

## Download and unzip Datasets
- Download needed datasets from this link: [ODDatasets](https://drive.google.com/drive/folders/1-jHkZweZC0nJPUZcutzJqoRxL-Yvz57q?usp=drive_link).
- Unzip the ODDatasets.zip file.
- All the datasets are stored as 'h5ad', and read by [Scanpy](https://scanpy.readthedocs.io/en/stable/).
  
|Dataset|Data Type|Observation|Feature|Outliers|Outlier Type|
|---|---|---|---|---|---|
|PBMC_full_ref|scRNA-seq| | |0%|Nan|
|PBMC_full_B|scRNA-seq| | |0|B cells|
