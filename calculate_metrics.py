import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataclasses import dataclass
from typing import List

@dataclass
class ImagePaths:
    paths1: List[str]
    paths2: List[str]
    unique_images: List[str]
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
    pretrained_dict = torch.load(model_path, weights_only=False)
    
    # Remove 'module.' prefix if present
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def optimal_threshold(distances, ground_truth):
    """
    Calculate the optimal threshold for a binary classification problem using the ROC curve and Youden's J statistic.

    Parameters:
    distances (list or numpy array): The predicted distances or probabilities for the positive class.
    ground_truth (list or numpy array): The ground truth binary labels (0 or 1).

    Returns:
    float: The optimal threshold value that maximizes Youden's J statistic.

    Notes:
    - The function assumes that the positive class is labeled as 0.
    - The ROC curve is calculated using the `roc_curve` function from the `sklearn.metrics` module.
    - Youden's J statistic is defined as `tpr - fpr`, where `tpr` is the true positive rate and `fpr` is the false positive rate.
    """
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve( (np.array(ground_truth).astype(int)), distances, pos_label=0)

    # Calculate Youden's J statistic
    youden_j = tpr - fpr

    # Find the optimal threshold
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# Calculate accuracy given a threshold using numpy
def calculate_accuracy(distances, ground_truth, threshold):
    """Calculates the accuracy given a threshold."""
    predictions = (distances < threshold).astype(int)
    accuracy = (predictions == ground_truth).astype(float).mean()
    
    return accuracy

def calculate_kfold_accuracy(distances, ground_truth):
    accuracies = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    result = pd.DataFrame()
    for test_index, train_index  in kf.split(distances):
        distances_train = distances[train_index]
        ground_truth_train = ground_truth[train_index]
        distances_test = distances[test_index]
        ground_truth_test = ground_truth[test_index]
        threshold = optimal_threshold(distances_train, ground_truth_train)
        accuracy = calculate_accuracy(distances_test, ground_truth_test, threshold)
        
        accuracies.append(accuracy)
    return np.mean(accuracies)

def get_rfw_paths(df):
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
    return ImagePaths(paths1=paths1, paths2=paths2, unique_images=unique_images)

def get_lfw_paths(df):
    # Collect all image paths and pair mappings
    unique_images = set()
    paths1 = []
    paths2 = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Collecting paths'):
        img1 = row['img_1']
        img2 = row['img_2']
        
        # Generate paths
        path1 = os.path.join('./data/imgs_', img1)
        
        path2 = os.path.join('./data/imgs_', img2)
        
        unique_images.update([path1, path2])
        paths1.append(path1)
        paths2.append(path2)

    # Batch process all unique images
    unique_images = list(unique_images)
    return ImagePaths(paths1=paths1, paths2=paths2, unique_images=unique_images)



def get_distances_from_paths(imagePaths, transform, model):
    dataset = ImageDataset(imagePaths.unique_images, transform)
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
    embeddings1 = [embedding_cache[path] for path in imagePaths.paths1]
    embeddings2 = [embedding_cache[path] for path in imagePaths.paths2]
    
    distances = torch.linalg.vector_norm(
        torch.stack(embeddings1) - torch.stack(embeddings2),
        dim=1
    ).numpy()

    return distances

# The rest of the functions remain the same except for removing get_embedding
# and modifying main() to remove normalization if not needed
def calculate_for_rfw(checkpoint_path):
    model = load_model_from_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    df = pd.read_csv('./data/RFW/rfw.csv')
    imagePaths = get_rfw_paths(df)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    distances = get_distances_from_paths(imagePaths, transform, model)
    print(f'\n The calculated accuracy for RFW is : {calculate_kfold_accuracy(distances, df.y_true)}')
    df['dist'] = distances
    return distances, df

def calculate_for_lfw(checkpoint_path):
    model = load_model_from_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    df = pd.read_csv('./lfw_test_pair.txt', sep = ' ')
    imagePahts = get_lfw_paths(df)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    distances = get_distances_from_paths(imagePahts, transform, model)
    print(f'\n The calculated accuracy for LFW is : {calculate_kfold_accuracy(distances, df.y_true)}')
    df['dist'] = distances
    return distances, df


def main():
    model_path = 'checkpoints/resnet18_25.pth'
    calculate_for_lfw(model_path)
    distances, df = calculate_for_rfw(model_path)
    
    df['dist'] = distances
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df[['img_1', 'img_2', 'dist']].to_csv(f'results_arcface_{timestamp}.csv', index=False)

if __name__ == '__main__':
    main()
