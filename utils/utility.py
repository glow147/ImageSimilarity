import yaml
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

import torchvision.transforms as transforms

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def denormalize(img):
    invTrans = transforms.Compose([ 
        transforms.Normalize(mean = [ 0., 0., 0. ],
                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                    std = [ 1., 1., 1. ]),
        ])
    
    return invTrans(img)

# Cosine Similarity를 구합니다.
@torch.no_grad()
def get_CosineSimilarity(embeddings):
    outputs = nn.functional.normalize(embeddings)

    cosine_similarity = outputs @ outputs.T

    cosine_similarity = cosine_similarity.fill_diagonal_(0)

    return cosine_similarity

# 가장 유사한 샘플 이미지를 생성합니다.
def get_SampleImage(cosine_similarity, anchor, random_indices):
    images = []

    for i in random_indices:
        max_index = torch.argmax(cosine_similarity[i]).item()
        images.extend([denormalize(anchor[i]), denormalize(anchor[max_index])])

    grid = make_grid(images, nrow=2)
    
    return grid

# Similarity가 0.9 이상인 이미지 페어를 만듭니다.
def get_Pairs(similarity):
    pairs = []

    for i in range(len(similarity)):
        max_index = torch.argmax(similarity[i]).item()
        if similarity[i][max_index].item() >= 0.9:
            pairs.append([i,max_index,similarity[i][max_index].item()])

    return pairs