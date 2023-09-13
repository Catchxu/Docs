import docs
import torch


input_dir = '/volume3/kxu/KDD_data/Pretrain_PBMC/'
# n_gene = ['3000', '6000', 'full']
n_gene = ['3000']
pretrain_list = [f'Pretrain_{i}' for i in n_gene]
# weight_index = ['Batch_' + i for i in n_gene] + ['Instance_' + i for i in n_gene]
weight_index = ['Batch', 'Instance']
weight_list = []
net = 'Conv1d'

for norm in ['Batch', 'Instance']:
    for name in pretrain_list:
        print(f'Train {norm} normalization model on dataset {name}:')
        adata = docs.read(input_dir, name)
        w = docs.pretrain(adata, norm_type=norm, net_type=net)
        weight_list.append(w)

state = dict(zip(weight_index, weight_list))
torch.save(state, f'model_{net}.pth')