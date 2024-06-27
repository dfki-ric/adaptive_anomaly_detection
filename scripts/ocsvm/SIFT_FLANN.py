# Sift & Flann method training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import img_to_array
import keras
import tensorflow as tf
import glob
import numpy as np
import cv2
import torch.nn as nn

# enabling gpu
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class sift_flann_training():
    def __init__(self,model,train_datapath,percent_similarity,train_image_features):
        self.model = model
        self.train_datapath = train_datapath
        self.percent_similarity = percent_similarity
        self.layer_name = 'block6h_add'
        self.intermediate_layer_modelb4 = keras.Model(inputs=self.model.input,
                                                outputs=self.model.get_layer(self.layer_name).output)
    
    # finds similarity between test image and training dataset using feature comparison with sift and flann method
    def similarity_block(self,test_image):
        original = cv2.imread(test_image)
        data_saved = []

        sift = cv2.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)

        # Define the FLANN object here
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(check=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        titles_compared = []
        train_data_compare = []
        titles = []
        ALLOWED_EXTENSIONS = [".jpg", ".png",'.JPG']
        train_images = [img for img in [glob.glob(self.train_datapath +'/*'+ext)for ext in ALLOWED_EXTENSIONS]]
        #load all the train image 
        for train_img in train_images:
            if len(train_img) !=0:
                for k in train_img:
                    image = cv2.imread(k)
                    titles.append(k)
                    train_data_compare.append(image)

        #feature matching with FLANN
        titles_compared = []
        for image_to_compare, title in zip(train_data_compare, titles):
            # 1) Check if 2 images are equals 
            if original.shape == image_to_compare.shape:
       
                difference = cv2.subtract(original, image_to_compare)
                b, g, r = cv2.split(difference)

                if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                    print("Similarity: 100% (equal size and channels)")
                    break

            # 2) Check for similarities between the 2 images
            kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

            matches = flann.knnMatch(desc_1, desc_2, k=2)
            matchesMask = [[0,0] for i in range(len(matches))]
            good_points = []
            for i,(m,n) in enumerate(matches):
                if m.distance < self.percent_similarity*n.distance:
                    good_points.append(m)

            number_keypoints = 0
            if len(kp_1) >= len(kp_2):
                number_keypoints = len(kp_1)
            else:
                number_keypoints = len(kp_2)

            percentage_similarity = (len(good_points)/ number_keypoints * 100)
           
            if percentage_similarity>3:           #texture = 40 and for solid 65/70 percentage
                titles_compared.append(title)

            
        n = len(train_data_compare) - len(titles_compared)
        d = len(train_data_compare)
        cal = n / d * 100
        data_saved.append(cal)
        print(cal, "% data saved from unwanted training")
        return titles_compared, cal
    
