#Normality model for anomaly detection 
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gmm import GaussianMixture


def mahalanobis(u, v, inv_cov):
    delta = u - v  # delta: shape (batch_size, num_vectors, num_features)    
    # Reshape delta to (batch_size * num_vectors, num_features) for batch matrix multiplication
    delta_reshaped = delta.view(-1, delta.shape[-1])    
    # Compute (delta @ inv_cov) for each pair
    delta_inv_cov = torch.matmul(delta_reshaped, inv_cov)    
    # Compute Mahalanobis distance: sqrt(delta @ inv_cov @ delta.T)
    # mahalanobis_squared = torch.einsum('ij,ij->i', delta_inv_cov, delta_reshaped)
    mahalanobis_squared = torch.sum(delta_inv_cov * delta, dim=-1)
    # Reshape the result back to (batch_size, num_vectors)
    mahalanobis_distances = torch.sqrt(mahalanobis_squared.view(delta.shape[0], delta.shape[1]))    
    return mahalanobis_distances

def detection_train_block(titles_compared,train_image_features):
    train_val = []
    for train_img in titles_compared: 
            train_val.append(train_image_features[train_img])
    train_images = torch.stack(train_val,dim=0)
    features_linear = train_images.reshape(-1, train_val[0].shape[-1])

        #Gaussian mixture model
    gaussian_model = GaussianMixture(1, 272).cuda()

    _, log_resp = gaussian_model._e_step(features_linear)
    pi, mu, var = gaussian_model._m_step(features_linear, log_resp)
    inv_con_model = torch.inverse(var)

    dist_list=[]
    for train_img in titles_compared: 
        features_t = train_image_features[train_img]
        featurest_linear_tensor = features_t.view(-1, features_t.shape[-1])
        new_dis = mahalanobis(mu,featurest_linear_tensor,inv_con_model)
        dist_list.append(new_dis.max().data.cpu().numpy())

    dist_list = np.array(dist_list)
    threshold = dist_list.max()
    return mu, inv_con_model, threshold

def normalize(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min)

def detection_test_block(features,mu,inv_con_model,threshold,test_image):
    features_linear = features.reshape(-1, features.shape[-1])
    features_linear = torch.from_numpy(features_linear).cuda()
    maha = mahalanobis(mu,features_linear,inv_con_model)

    if threshold <= maha.max():
        print("Result : Anomaly")
        score = maha.reshape(1, 1, 7, 7)
        score = nn.functional.interpolate(score, size=[224, 224], mode="bicubic", align_corners=True).squeeze()
        normalize_score = normalize(score)
        mask_threshold = [torch.max(normalize_score[n]).data.cpu() for n in range(len(normalize_score))]
        mask = normalize_score > np.mean(mask_threshold)
        image_mask = test_image
        alpha = 0.45
        mask = mask.data.cpu()
        image_mask[mask] = image_mask[mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        result_image = image_mask
        anomaly_prediction = 1


    else:
        print("Result : No Anomaly")
        result_image = test_image
        anomaly_prediction = 0


    mu = mu.cpu()
    inv_con_model = inv_con_model.cpu()
    features_linear = features_linear.cpu()
    del mu, inv_con_model, features_linear
    torch.cuda.empty_cache()
    return result_image, anomaly_prediction
         

         

     

