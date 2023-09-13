import warnings
import pandas as pd
import scanpy as sc
import anndata as ad
from math import e
from typing import Literal


def read(data_dir: str, data_name: str, preprocess: bool = True,
         return_type: Literal['anndata', 'pandas'] = 'anndata'):
    input_dir = data_dir + data_name + '.h5ad'
    adata = sc.read(input_dir)

    if preprocess:
        adata = preprocess_data(adata)
    
    if return_type == 'anndata':
        return adata
    elif return_type == 'pandas':
        x = pd.DataFrame(
            adata.X, index=adata.obs_names, columns=adata.var_names)
        y = pd.DataFrame(
            adata.obs['cell.type'], index=adata.obs_names, columns=['cell_type'])
        return x, y


def read_cross(ref_dir: str, tgt_dir:str, ref_name: str, tgt_name: str, 
               preprocess: bool = True, n_genes: int = 3000,
               return_type: Literal['anndata', 'pandas'] = 'anndata'):
    ref = read(ref_dir, ref_name, preprocess=False)
    tgt = read(tgt_dir, tgt_name, preprocess=False)
    overlap_gene = list(set(ref.var_names) & set(tgt.var_names))
    ref = ref[:, overlap_gene]
    tgt = tgt[:, overlap_gene]

    if preprocess:
        ref = preprocess_data(ref)
        tgt = preprocess_data(tgt)
        if len(overlap_gene) <= n_genes:
            warnings.warn(
                'There are too few overlapping genes to perform feature selection'
            )
        else:
            sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
            tgt = tgt[:, ref.var_names]

    if return_type == 'anndata':
        return ref, tgt
    elif return_type == 'pandas':
        ref_x = pd.DataFrame(
            ref.X, index=ref.obs_names, columns=ref.var_names)
        ref_y = pd.DataFrame(
            ref.obs['cell.type'], index=ref.obs_names, columns=['cell_type'])
        tgt_x = pd.DataFrame(
            tgt.X, index=tgt.obs_names, columns=tgt.var_names)
        tgt_y = pd.DataFrame(
            tgt.obs['cell.type'], index=tgt.obs_names, columns=['cell_type'])
        return ref_x, ref_y, tgt_x, tgt_y


def preprocess_data(adata: ad.AnnData):
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata


if __name__ == '__main__':
    data_dir = '/volume3/kxu/KDD_data/PBMC/'
    data_name = 'PBMC_3000_ref'
    read(data_dir, data_name)
    read(data_dir, data_name, return_type='pandas')

    ref_dir = '/volume3/kxu/KDD_data/PBMC/'
    ref_name = 'PBMC_full_ref'
    tgt_dir = '/volume3/kxu/KDD_data/TME/'
    tgt_name = 'TME_full_Tumor'
    read_cross(ref_dir, tgt_dir, ref_name, tgt_name)