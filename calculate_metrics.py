
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models import *
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_curve
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import datetime


def get_embedding(img_path, transform, model):
    pil_image = Image.open(img_path)
    pil_image = transform(pil_image)
    embeding = model(pil_image.unsqueeze(0))
    # print(embeding.shape)
    return embeding

def get_random_image(classes, validation_folder):
    random_class = np.random.choice(classes)
    class_folder = os.path.join(validation_folder, random_class)
    images = os.listdir(class_folder)
    random_image = np.random.choice(images)
    image_path = os.path.join(class_folder, random_image)
    return image_path, random_class

def get_random_image_from_class(class_folder):
    images = os.listdir(class_folder)
    random_image = np.random.choice(images)
    image_path = os.path.join(class_folder, random_image)
    return image_path

def l2_distance_torch(embedding1, embedding2):
    """Computes the L2 (Euclidean) distance between two embeddings."""
    return torch.linalg.vector_norm(embedding1 - embedding2).item()

def get_different_pair_distances(validation_folder, transform, model, n):
    classes = os.listdir(validation_folder)
    distances = []
    cnt = 0
    seen_pairs = set()
    print('Getting different pair distances')
    with tqdm(total=n) as pbar:
        while cnt < n:
            # Select two random images
            image_path_1, class_1 = get_random_image(classes, validation_folder)
            image_path_2, class_2 = get_random_image(classes, validation_folder)
            # Check that the classes are different
            if class_1 == class_2 or (image_path_1, image_path_2) in seen_pairs:
                continue
            seen_pairs.add((image_path_1, image_path_2))
            seen_pairs.add((image_path_2, image_path_1))

            embedding_1 = get_embedding(image_path_1, transform, model)
            embedding_2 = get_embedding(image_path_2, transform, model)
            distances.append(l2_distance_torch(embedding_1, embedding_2))
            cnt += 1
            pbar.update(1)
    return distances

def get_same_pair_distances(validation_folder, transform, model, n):
    classes = os.listdir(validation_folder)
    distances = []
    cnt = 0
    seen_pairs = set()

    print('Getting same pair distances')
    with tqdm(total=n) as pbar:
        while cnt < n:
            image_path_1, class_1 = get_random_image(classes, validation_folder)
            image_path_2 = get_random_image_from_class(os.path.join(validation_folder, class_1))

            # Check the images are different
            if (image_path_1, image_path_2) in seen_pairs or image_path_1 == image_path_2:
                continue
            seen_pairs.add((image_path_1, image_path_2))
            seen_pairs.add((image_path_2, image_path_1))

            embedding_1 = get_embedding(image_path_1, transform, model)
            embedding_2 = get_embedding(image_path_2, transform, model)

            distances.append(l2_distance_torch(embedding_1, embedding_2))
            cnt += 1
            pbar.update(1)
    return distances


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
    
# Calculate accuracy given a threshold using numpy
def calculate_accuracy(distances, ground_truth, threshold):
    """Calculates the accuracy given a threshold."""
    predictions = (distances < threshold).astype(int)
    accuracy = (predictions == ground_truth).astype(float).mean()
    
    return accuracy

def calculate_kfold_accuracy(distances, ground_truth, df):
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
        
        df_test = df.iloc[test_index].copy()
        df_test['preds'] = (distances_test < threshold).astype(int)


        result = pd.concat([result, df_test], ignore_index=True)
        accuracies.append(accuracy)
    return np.mean(accuracies), result

def get_distances_from_df(df, transform, model):
	distances = []
	# embeddings_cache = {}

	for i in tqdm(range(len(df))):
		image_path_1 = f'./data/RFW/aligned_imgs/{'_'.join(df.iloc[i].img_1.split('_')[0:-1])}-{df.iloc[i].ethnicity.split(' ')[0]}/{df.iloc[i].img_1}'
		image_path_2 = f'./data/RFW/aligned_imgs/{'_'.join(df.iloc[i].img_2.split('_')[0:-1])}-{df.iloc[i].ethnicity.split(' ')[-1]}/{df.iloc[i].img_2}'
		embedding_2 = F.normalize(get_embedding(image_path_2, transform, model))
		embedding_1 = F.normalize(get_embedding(image_path_1, transform, model))
		distances.append(l2_distance_torch(embedding_1, embedding_2))
	return distances

def main():
    np.random.seed(88)
    model_path = 'checkpoints/resnet18_99.pth'
    model = load_model_from_checkpoint(model_path)
    validation_folder = 'data/imgs_subset_complete/validation'

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    df = pd.read_csv('./data/RFW/rfw.csv')
    distances = get_distances_from_df(df, transform, model)
    df['dist'] = distances
    # df['dist'] = df['dist'] / df['dist'].mean()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df[['img_1', 'img_2', 'dist']].to_csv(f'../FaVFA/model_results/results_arcface_{timestamp}.csv', index=False)

if __name__ == '__main__':
    main()
