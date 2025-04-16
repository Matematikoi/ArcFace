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
import ast
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    result = []
    # Collect all image paths and pair mappings
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Collecting paths'):
        img = row['img_id']
        ethnicity = row['ethnicity']
        
        # Generate paths
        dir_part = '_'.join(img.split('_')[:-1]) + '-' + ethnicity
        path = os.path.join('./data/RFW/aligned_imgs', dir_part, img)
        result.append(path)
    return result


def get_distances_from_paths(imagePaths, transform, model):
    dataset = ImageDataset(imagePaths, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=os.cpu_count(), 
        pin_memory=True,
    )

    # Cache embeddings
    embedding_cache = {}
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing images'):
            if not batch:  # Skip empty batches
                continue

            batch_paths, batch_images = batch

            # If batch_images is a list, stack it into a single tensor.
            if isinstance(batch_images, list):
                if len(batch_images) == 0:
                    continue
                batch_images = torch.stack(batch_images)

            # Always move the batch to the same device as the model.
            batch_images = batch_images.to(device)

            batch_embeddings = model(batch_images).cpu()
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=1)

            for path, embedding in zip(batch_paths, batch_embeddings):
                embedding_cache[path] = embedding.numpy()

    return [embedding_cache[path] for path in imagePaths]

# The rest of the functions remain the same except for removing get_embedding
# and modifying main() to remove normalization if not needed
def calculate_for_rfw(checkpoint_path):
    model = load_model_from_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    df = pd.read_csv('./data/RFW/rfw_by_demography.csv')
    imagePaths = get_rfw_paths(df)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    df['embeddings'] = get_distances_from_paths(imagePaths, transform, model)
    return df


def main():
    model_path = 'checkpoints_arcface_70acc_lambda/resnet18_99.pth'
    data = calculate_for_rfw(model_path)
    
    print(data.head())
    # Convert embeddings from string to list if needed
    data['embeddings'] = data['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

    # Extract embeddings and labels
    embeddings = np.vstack(data['embeddings'].values)
    gender_labels = data['gender'].values
    ethnicity_labels = data['ethnicity'].values

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500, learning_rate=200, method='barnes_hut', angle=0.5)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot function
    def plot_tsne(tsne_results, labels, title, filename, alpha=0.05):
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        scatter_objects = []
        colors = plt.cm.get_cmap('Set1', len(unique_labels))
        for i, label in enumerate(unique_labels):
            idx = labels == label
            scatter = plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=label, alpha=alpha, color=colors(i))
            scatter_objects.append(scatter)
        
        # Create opaque legend markers
        legend_markers = [plt.Line2D([0], [0], marker='o', color=colors(i), linestyle='None', markersize=8) for i in range(len(unique_labels))]
        plt.legend(legend_markers, unique_labels, loc='best')
        
        plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.savefig(filename)
        plt.close()
    # Save t-SNE for gender
    plot_tsne(tsne_results, gender_labels, "t-SNE Visualization by Gender", "tsne_gender.png")

    # Save t-SNE for ethnicity
    plot_tsne(tsne_results, ethnicity_labels, "t-SNE Visualization by Ethnicity", "tsne_ethnicity.png")

if __name__ == '__main__':
    main()
