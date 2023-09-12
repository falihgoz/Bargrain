import argparse
import torch
import numpy as np
import os
import omegaconf
from src.model import Bargrain
from sklearn.model_selection import train_test_split
from src.data import *
from torch_geometric.loader import DataLoader
from src.train import train, test_model
# import wandb

dataset_base_folder = 'preprocessed_data'

def run(dataset_name, config, device):
    dataset_folder = os.path.join(dataset_base_folder, dataset_name)
    dataset = np.load(f'{dataset_folder}/{dataset_name}.npy',
                      allow_pickle=True).item()
    
    num_nodes = dataset['corr_graph'].shape[1]
    tc_length = dataset['tc'].shape[2]
    indices = list(range(dataset['labels'].shape[0]))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=dataset['labels'], random_state=0)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15, stratify=dataset['labels'][train_val_idx], random_state=0)

    print(f'Train Len: {len(train_idx)}')
    print(f'Val Len: {len(val_idx)}')
    print(f'Test Len: {len(test_idx)}')

    train_data = Brain(train_idx, dataset, config.th)
    val_data = Brain(val_idx, dataset, config.th)
    test_data = Brain(test_idx, dataset,  config.th)
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    model = Bargrain(num_nodes, config, device, tc_length)

    save = f'./checkpoints/Bargrain-{args.dataset}.pt'
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')
    
    trained_model = train(model, train_loader, val_loader, config, device, save)
    acc, sens, spec, auc = test_model(test_loader, device, save)

    print(" Test Accuracy = {:.2f}% \n Test Sens = {:.2f}% \n Test Spec = {:.2f}% \n Test AUC = {:.2f}%".format(
        acc*100, sens * 100, spec*100, auc*100))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dataset', type=str, default='cobre',
                        help='Dataset name, valid options are ["abide", "cobre", "acpi"]')
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(".\conf\config.yaml")

    device = torch.device(args.device)
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    dataset_name = args.dataset

    run(dataset_name, config, device)