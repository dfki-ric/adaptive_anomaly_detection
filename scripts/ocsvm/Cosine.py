#Cosine similarity method training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
import keras
from sklearn.metrics.pairwise import cosine_similarity
import time
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

#enabling gpu
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class cosine_training():
    def __init__(self,model,train_datapath,percent_similarity,train_image_features):
        self.model = model
        self.train_datapath = train_datapath
        self.percent_similarity = percent_similarity
        self.layer_name = 'block6h_add'
        self.intermediate_layer_modelb4 = keras.Model(inputs=self.model.input,
                                                outputs=self.model.get_layer(self.layer_name).output)
        
        # Initialize a dictionary to store titles and computed features for training images
        self.train_image_features = train_image_features

    
    

    # finds similarity between test image and training dataset using feature comparison with cosine method

    def similarity_block(self,test_image):
        #self.test_image = test_image
        img_test = cv2.imread(test_image)
        img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        img_test = cv2.resize(img_test, (224,224))
        img_test = img_to_array(img_test)
        img_test = np.expand_dims(img_test,axis=0)
        img_test = preprocess_input(img_test)

        feature_vector = self.intermediate_layer_modelb4.predict(img_test)
        a, b, c, n = feature_vector.shape
        feature_vector = feature_vector.reshape(b*c,n)

        titles_compared = [] 

        for train_img_title, train_features in self.train_image_features.items():
            a, b, c, n = self.train_image_features[train_img_title].shape
            train_feature_vector= self.train_image_features[train_img_title].reshape(b*c,n)
            image_similarity_cosine = cosine_similarity(feature_vector,train_feature_vector)

            if image_similarity_cosine[0][0]>= self.percent_similarity:
                titles_compared.append(train_img_title)

        
        n = len(self.train_image_features) - len(titles_compared)
        d = len(self.train_image_features)
        cal = n / d * 100
        print(cal, "% data saved from unwanted training")
        return titles_compared , cal
