# OutlierDetection
Outlier detection model for KDD2024

## Download and unzip Datasets
- Download needed datasets from this link: [ODDatasets](https://drive.google.com/drive/folders/1-jHkZweZC0nJPUZcutzJqoRxL-Yvz57q?usp=drive_link).
- Unzip the ODDatasets.zip file.
- All the datasets are stored as 'h5ad', and read by [Scanpy](https://scanpy.readthedocs.io/en/stable/).
  
|Dataset|Data Type|Observation|Feature|Sparsity|Outliers|Outlier Type|
|---|---|---|---|---|---|---|
|PBMC_3000_ref|scRNA-seq|4698|3000|98.83%|0.00%|Nan|
|PBMC_3000_B|scRNA-seq|3684|3000|98.85%|13.14%|B cells|
|PBMC_3000_NK|scRNA-seq|3253|3000|98.71%|12.73%|NK cells|
|PBMC_10000_ref|scRNA-seq|4698|10000|98.92%|0.00%|Nan|
|PBMC_10000_B|scRNA-seq|3684|10000|98.92%|13.14%|B cells|
|PBMC_10000_NK|scRNA-seq|3253|10000|98.82%|12.73%|NK cells|
|PBMC_full_ref|scRNA-seq|4698|32738|98.82%|0.00%|Nan|
|PBMC_full_B|scRNA-seq|3684|32738|98.83%|13.14%|B cells|
|PBMC_full_NK|scRNA-seq|3253|32738|98.71%|12.73%|NK cells|
|Cancer_3000_ref|scRNA-seq|8104|3000|92.94%|0.00%|Nan|
|Cancer_3000_EI|scRNA-seq|7721|3000|94.74%|50.03%|Epithelial Tumor & Immune Tumor|
|Cancer_3000_ES|scRNA-seq|4950|3000|93.11%|58.12%|Epithelial Tumor & Stromal Tumor|
|Cancer_10000_ref|scRNA-seq|8104|10000|93.02%|0.00%|Nan|
|Cancer_10000_EI|scRNA-seq|7721|10000|94.79%|50.03%|Epithelial Tumor & Immune Tumor|
|Cancer_10000_ES|scRNA-seq|4950|10000|93.22%|58.12%|Epithelial Tumor & Stromal Tumor|
|Cancer_full_ref|scRNA-seq|8104|33538|92.90%|0.00%|Nan|
|Cancer_full_EI|scRNA-seq|7721|33538|94.68%|50.03%|Epithelial Tumor & Immune Tumor|
|Cancer_full_ES|scRNA-seq|4950|33538|93.15%|58.12%|Epithelial Tumor & Stromal Tumor|
