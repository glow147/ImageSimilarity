import torch
import random
import argparse
import lightning as L
import torchvision.transforms as transforms

from pathlib import Path
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from models.model import SimilarityModel
from utils.datasets import CustomDataset, CustomTestDataset
from utils.utility import *

from time import time
from datetime import datetime

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.set_float32_matmul_precision('medium')
random.seed(0)

def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['weight'] is None:
        raise "Please Input weight Path"
    total_start = time()
    print('Data Loading Start')
    data_loading_start = time()
    if config['load_from_disk']:
        try:
            dataset = load_from_disk('tmp')
        except:
            print(f"Error loading dataset from disk. Loading from hugging face '{config['dataset']}' instead.")
            dataset = load_dataset(config['dataset'])
    else:
        dataset = load_dataset(config['dataset'])
    data_loading_end = time()
    print(f'Data Loading Done!')

    print('Model Loading Start')
    model_loading_start = time()
    params = torch.load(config['weight'], map_location=device)
    similarity_model = SimilarityModel(config)
    similarity_model.load_state_dict(params['state_dict'])
    similarity_model.eval()
    model_loading_end = time()
    print(f'Model Loading Done!')

    test_transforms = transforms.Compose(
        [
            transforms.Resize(config['DATA']['IMAGE_SIZE']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    test_dataset = CustomTestDataset(dataset['test'].select(range(1000)), test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=config['DATA']['BATCH_SIZE'],
                                  num_workers=config['DATA']['NUM_WORKERS'])
    cosine_similarity_start = time()
    print("Cosine Similarity Calcuation Start")
    embeddings = torch.Tensor([])
    with torch.no_grad():
        for i, images in enumerate(test_dataloader):
            outputs = similarity_model(images)
            embeddings = torch.cat((embeddings, outputs))
            print(f'\rGet Embedding Process {i+1}/{len(test_dataloader)} Done!',end='')
    cos_sim = get_CosineSimilarity(embeddings)

    pairs = get_Pairs(cos_sim)
    pairs.sort(key=lambda x:x[-1], reverse=True)
    cosine_similarity_end = time()
    print(f'\nCosine Similarity Calcuation Done!')

    with open('result.csv', 'w') as f:
        print('ImageA(Index),ImageB(Index),Similarity', file=f)
        for pair in pairs:
            print(f'{pair[0]},{pair[1]},{pair[2]:.3f}', file=f)
            if config['save_image']:
                file_name = f'{pair[0]},{pair[1]},{pair[2]:.3f}.PNG'
                imgA, imgB = test_dataset[pair[0]], test_dataset[pair[1]]
                grid = make_grid([denormalize(imgA), denormalize(imgB)], nrow=2)
                save_image(grid, file_name)

    print('-' * 20)
    print(f'Data Loading Time: {data_loading_end - data_loading_start:.4f}s')
    print(f'Model Loading Time: {model_loading_end - model_loading_start:.4f}s')
    print(f'Cosine Similarity Calcuation Time: {cosine_similarity_end - cosine_similarity_start:.4f}s')
    print(f'Total Execution Time: {time() - total_start:.4f}s')
    print(f'Number Of Similarity Image(Over 0.9) : {len(pairs)}')
    print('-' * 20)

def train(config):
    if config['load_from_disk']:
        try:
            dataset = load_from_disk('tmp')
        except:
            print(f"Error loading dataset from disk. Loading from hugging face '{config['dataset']}' instead.")
            dataset = load_dataset(config['dataset'])
    else:
        dataset = load_dataset(config['dataset'])

    if config['resume'] and config['weight'] is None:
        raise "Please Input weight path"
    
    Path(config['output_dir']).mkdir(exist_ok=True, parents=True)

    similarity_model = SimilarityModel(config)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(config['DATA']['IMAGE_SIZE']),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(config['DATA']['IMAGE_SIZE']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    train_dataset = CustomDataset(dataset['train'].select(range(200000)), train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=config['DATA']['BATCH_SIZE'],
                                  pin_memory=True,
                                  num_workers=config['DATA']['NUM_WORKERS'], shuffle=True)

    valid_dataset = CustomDataset(dataset['validation'].select(range(10000)), valid_transforms)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['DATA']['BATCH_SIZE'],
                                  num_workers=config['DATA']['NUM_WORKERS'])

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=config['output_dir'],
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        save_top_k=3,  
        mode='min'    
    )

    trainer = L.Trainer(accelerator='gpu', devices=torch.cuda.device_count(),
                        max_epochs=config['TRAIN']['EPOCHS'],
                        strategy='ddp',
                        precision='16-mixed', gradient_clip_algorithm='norm',
                        callbacks=[checkpoint_callback,])
    
    trainer.fit(similarity_model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=valid_dataloader,
                ckpt_path=config['weight'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Settings
    parser.add_argument("--configs", "-c", type=str, default="configs/default.yaml")
    
    # Training Settings
    parser.add_argument("--dataset", "-d", type=str, default="imagenet-1k")
    parser.add_argument("--loss", choices=['cos', 'L2'], default='cos')
    parser.add_argument("--output_dir", "-o", type=str, default="results")
    parser.add_argument("--save_image", "-s", action='store_true', default=False)
    parser.add_argument("--load_from_disk", "-l", action='store_false', default=True)
    parser.add_argument("--resume", "-r", action='store_true', default=False)
    parser.add_argument("--weight", "-w", type=str, default=None)
    
    # Testing Settings
    parser.add_argument("--test", "-t", action='store_true', default=False)

    args = parser.parse_args()
    config = load_config(args.configs)
    config.update(vars(args))
    print(config)
    procedure_name = 'Train' if not config['test'] else 'Test'
    print(f'Start {procedure_name} Procedure Time {datetime.now()}')
    if not config['test']:
        train(config)
    else:
        test(config)
    print(f'End {procedure_name} Procedure Time {datetime.now()}')
    
