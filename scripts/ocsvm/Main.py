#define libraries
import argparse
from ast import Str
import keras
import tensorflow as tf
import glob
import cv2 
import numpy as np 
import pandas as pd
from SIFT_FLANN import *
from Cosine import *
import torch.nn as nn
import os
import time
from tensorflow.keras.utils import img_to_array
import torch
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn import svm


def config():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-m','--anomaly_detection_method', type=str, choices=["c","sf"],
                            help="Cosine method or SIFT-FLANN method : ", required=True)
    
    parser.add_argument('-f','--pretrained_features', type=str, choices=["y","n"],help="Flag set to determine if the image features pretrained or not", required=True)

    parser.add_argument('-d','--train_data_path', type=str, help="Folder containing images('.jpg', '.png', '.JPG')", required=True)

    parser.add_argument('-t','--test_data_path', type=str, help="Folder containing images to tested", required=True)

    parser.add_argument('-r','--result_path', type=str, help="Path to save results", required=True)

    parser.add_argument('-v','--visualization_on_off', type=int,choices=[0,1], help="Visualization on/off 1 == 'on' and 0 == 'off'", default=0)


    parser.print_help()
    # Execute parse_args()
    args = parser.parse_args()

    return args

def normalize(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)

# Initialize a dictionary to store titles and computed features for training images
def precompute_train_features(train_datapath,intermediate_layer_modelb4):
        ALLOWED_EXTENSIONS = [".jpg", ".png", '.JPG']
        train_image_features ={}

        for ext in ALLOWED_EXTENSIONS:
            train_images = glob.glob(os.path.join(train_datapath, '*' + ext))
            for train_img in train_images:
                img = cv2.imread(train_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (224, 224))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img.repeat(3, axis=-1)
                img = preprocess_input(img)
                
                # Extract features using intermediate layer
                img_features = intermediate_layer_modelb4.predict(
                    img,
                    batch_size=None,
                    verbose=0,
                    steps=None,
                    callbacks=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                )
                # Store both titles and computed features in the dictionary
                train_image_features[train_img] = img_features
        return train_image_features

     
def main(cfg):
                                                        
    #create a path for storing results
    if not os.path.exists(os.path.join(cfg.result_path,"results")):
        os.makedirs(os.path.join(cfg.result_path,"results"))
    results_folder_path = os.path.join(cfg.result_path,"results")

    #create a path for storing heatmap within results
    if not os.path.exists(os.path.join(results_folder_path,"heatmap")):
        os.makedirs(os.path.join(results_folder_path,"heatmap"))
    heatmap_result_path = os.path.join(results_folder_path,"heatmap")

    model = tf.keras.applications.EfficientNetB4(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(224, 224, 3),
                pooling=None,
                )
    layer_name = 'block6h_add'
    intermediate_layer_modelb4 = keras.Model(inputs= model.input,
                                                outputs= model.get_layer(layer_name).output)
    clf =svm.OneClassSVM(gamma='scale',nu=0.1, kernel="rbf")
    
    # run similarity model to load features of only similar data images
    if cfg.pretrained_features == 'y':
        if cfg.anomaly_detection_method =='c':
            percent_similarity = 0.85
            train_image_features = torch.load(os.path.join(os.path.join(cfg.train_data_path,'train_weights_path'), 'train_image_features.pth'), map_location=torch.device('cpu'))
            online_training_int = cosine_training(model, cfg.train_data_path,percent_similarity,train_image_features)
            print("-----------------features loaded-------------------")
        elif cfg.anomaly_detection_method =='sf':
            percent_similarity = 0.7
            train_image_features = torch.load(os.path.join(os.path.join(cfg.train_data_path,'train_weights_path'), 'train_image_features.pth'), map_location=torch.device('cpu'))
            online_training_int = sift_flann_training(model, cfg.train_data_path,percent_similarity,train_image_features)
            print("-----------------features loaded-------------------")
    elif cfg.pretrained_features == 'n':
        if cfg.anomaly_detection_method =='c':
            percent_similarity = 0.85
            train_image_features = precompute_train_features(cfg.train_data_path,intermediate_layer_modelb4)
            online_training_int = cosine_training(model, cfg.train_data_path,percent_similarity,train_image_features)
            if not os.path.isdir(os.path.join(cfg.train_data_path,'train_weights_path')):
                os.mkdir(os.path.join(cfg.train_data_path,'train_weights_path'))
            torch.save(train_image_features, os.path.join(os.path.join(cfg.train_data_path,'train_weights_path'), 'train_image_features.pth'))
            print("-----------------calculate and store freatures-------------------")
        elif cfg.anomaly_detection_method =='sf':
            percent_similarity = 0.7
            train_image_features = precompute_train_features(cfg.train_data_path,intermediate_layer_modelb4)
            online_training_int = sift_flann_training(model, cfg.train_data_path,percent_similarity,train_image_features)
            if not os.path.isdir(os.path.join(cfg.train_data_path,'train_weights_path')):
                os.mkdir(os.path.join(cfg.train_data_path,'train_weights_path'))
            torch.save(train_image_features, os.path.join(os.path.join(cfg.train_data_path,'train_weights_path'), 'train_image_features.pth'))
            print("-----------------calculate and store freatures-------------------")
            
    #variable declaration
    predicted_value = []
    predicted_tag = []
    data_saved = []
    process_time = []
    image_counter = 1 

    #load test image from test datapath
    ALLOWED_EXTENSIONS = [".jpg", ".png", '.JPG']
    filenames = [img for img in [glob.glob(cfg.test_data_path + '/*'+ext)for ext in ALLOWED_EXTENSIONS]]
    fig  = plt.figure()

    for test_img in filenames:
        test_img.sort()
        for img in test_img:

            start_time = time.time()
            
            read_image = cv2.imread(img)
            img1 = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
            img1 = cv2.resize(img1, (224, 224))
            img1 = img_to_array(img1)
            img1 = img1.repeat(3, axis=-1)
            img2 = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (224, 224))
            print('test image counter', image_counter)
            #check the similarity
            titles_compared, cal = online_training_int.similarity_block(img)
            data_saved.append(cal)
            #fig, ax = plt.subplots()
            if len(titles_compared)!=0:
                #training the normality model for only selected similar images
                # calculate one-class-svm
                train_val = [train_image_features[train_img] for train_img in titles_compared]
                train_images = np.array(train_val)
                features_linear = train_images.reshape(-1, train_images.shape[-1])

                svm_one_class = clf.fit(features_linear)

                dist_list = []
                for train_img in titles_compared:
                    features2 = train_image_features[train_img]
                    features2_linear = features2.reshape(-1, features2.shape[-1])
                    maha = svm_one_class.decision_function(features2_linear)
                    dist_list.append(max(abs(maha)))

                dist_list = np.array(dist_list)
                threshold = dist_list.max()  #normality threshold


                #feature extraction per test image
                tes = np.array(img1)
                tes = np.expand_dims(tes, axis= 0)

                # calculate decision value on test image 
                features = intermediate_layer_modelb4.predict(tes)
                features_linear = features.reshape(-1, features.shape[-1])
                decision_score = svm_one_class.decision_function(features_linear)
                max_decision_score = max(abs(decision_score))
               
                if threshold <= max_decision_score:
                        print("Result : Anomaly")
                        predicted_value.append(1)
                        predicted_tag.append('Anomaly')
                        src = decision_score.reshape(1, 1, 7, 7)
                        src = torch.from_numpy(src)
                        src = nn.functional.interpolate(src, size=[224,224], mode="bilinear", align_corners=True).squeeze()
                        src = normalize(src)

                        #calculate score threshold for each image 
                        mask_threshold = []
                        for n in range(len(src)):
                            mask_threshold.append(torch.min(src[n]))
                        
                        mask = src < np.mean(mask_threshold)
                        #mask = normalize_score > 0.4
                        image_mask = img2
                        alpha = 0.45
                        image_mask[mask] = image_mask[mask] * (1-alpha) + np.array([255, 0, 0]) * alpha
                        image_plot = image_mask
                else:
                        print("Result : No Anomaly")
                        predicted_value.append(0)
                        predicted_tag.append('No Anomaly')
                        image_plot = img2
            else:
                print('No similar image found,test image does not match the environment')
                print("Result : Anomaly")
                predicted_value.append(0)
                predicted_tag.append('Anomaly')
                image_plot = img2
            
            if cfg.visualization_on_off == 1:
                plt.axis('off')
                plt.imshow(image_plot)
                plt.savefig((heatmap_result_path + '/{}').format(os.path.basename(img)))
                plt.show(block = False)
                plt.pause(0.5)
                fig.canvas.flush_events()
            else:
                plt.axis('off')
                plt.imshow(image_plot)
                plt.savefig((heatmap_result_path + '/{}').format(os.path.basename(img)))
            print("Computation time per frame: ", time.time() - start_time, "secs")
            process_time.append(time.time() - start_time)
            image_counter +=1
        plt.close()

                
    #calculating overall results
    avg_data = np.average(data_saved)
    print('Average % data saved from training: ', avg_data,'%')

    avg_time = np.average(process_time)
    print('Average computation time per frame: ', avg_time, "secs")

    #save average time and data
    unique_filename = 'result.csv'
    result_filepath = os.path.join(results_folder_path, unique_filename)


    # Create a dictionary with the data
    data = {'avg % data saved from training': [f"{avg_data} %"], 'avg computation time per frame': [f'{avg_time} secs']}
    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
   
    # Write the DataFrame to the CSV file
    df.to_csv(result_filepath, index=False)

    print(f"Average data and time is saved to {result_filepath}")


    #generate csv to store prediction
    unique_filename = 'prediction.csv'
    result_filepath = os.path.join(results_folder_path, unique_filename)


    # Create a dictionary with the data
    data = {'predicted_values': predicted_value, 'predicted_tag': predicted_tag}

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
   
    # Write the DataFrame to the CSV file
    df.to_csv(result_filepath, index=False)

    print(f"Predictions saved to {result_filepath}")

if __name__ == '__main__':
    cfg = config()
    main(cfg)












            





















            




            




    




