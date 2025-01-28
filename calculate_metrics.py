import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models import *
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_curve
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import datetime

class ImageDataset(Dataset):
    """Dataset to load images in batches"""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            return path, image
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None, None

def load_model_from_checkpoint(model_path):
    # Load model from function in train.py
    model = resnet_face18(use_se=False)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    
    # Remove 'module.' prefix if present
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def get_distances_from_df(df, transform, model):
    # Collect all image paths and pair mappings
    unique_images = set()
    paths1 = []
    paths2 = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Collecting paths'):
        img1 = row['img_1']
        img2 = row['img_2']
        ethnicity = row['ethnicity']
        
        # Generate paths
        dir_part1 = '_'.join(img1.split('_')[:-1]) + '-' + ethnicity.split(' ')[0]
        path1 = os.path.join('./data/RFW/aligned_imgs', dir_part1, img1)
        
        dir_part2 = '_'.join(img2.split('_')[:-1]) + '-' + ethnicity.split(' ')[-1]
        path2 = os.path.join('./data/RFW/aligned_imgs', dir_part2, img2)
        
        unique_images.update([path1, path2])
        paths1.append(path1)
        paths2.append(path2)

    # Batch process all unique images
    unique_images = list(unique_images)
    dataset = ImageDataset(unique_images, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=os.cpu_count(), 
        pin_memory=True,
        collate_fn=lambda x: [item for item in x if item[0] is not None]
    )

    # Cache embeddings
    embedding_cache = {}
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing images'):
            batch_paths, batch_images = zip(*batch)
            batch_images = torch.stack(batch_images).to(device)
            batch_embeddings = model(batch_images).cpu()
            
            for path, embedding in zip(batch_paths, batch_embeddings):
                embedding_cache[path] = embedding

    # Vectorized distance calculation
    embeddings1 = [embedding_cache[path] for path in paths1]
    embeddings2 = [embedding_cache[path] for path in paths2]
    
    distances = torch.linalg.vector_norm(
        torch.stack(embeddings1) - torch.stack(embeddings2),
        dim=1
    ).numpy()

    return distances

# The rest of the functions remain the same except for removing get_embedding
# and modifying main() to remove normalization if not needed

def main():
    np.random.seed(88)
    model = load_model_from_checkpoint('checkpoints/resnet18_99.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    df = pd.read_csv('./data/RFW/rfw.csv')
    distances = get_distances_from_df(df, transform, model)
    
    df['dist'] = distances
    # df['dist'] = df['dist'] / df['dist'].mean()  # Only if needed
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df[['img_1', 'img_2', 'dist']].to_csv(f'results_arcface_{timestamp}.csv', index=False)

if __name__ == '__main__':
    main()
