#import packages
#!/usr/bin/env python3
import os
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-11.8/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-11.8/nvvm/lib64/libnvvm.so"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import tensorflow as tf
import keras
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import time

from Cosine import *
from NN import *
from SIFT_FLANN import *

def configure_gpu(reduce_retracing=True):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True) 


class Config:
    def __init__(self,anomaly_detection_method, pretrained_features,train_data_path,test_data_path,result_path,visualization_on_off):
        self.anomaly_detection_method = anomaly_detection_method
        self.pretrained_features = pretrained_features
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.result_path = result_path
        self.visualization_on_off = visualization_on_off


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

    args = parser.parse_args()
    return Config(args.anomaly_detection_method, args.pretrained_features,args.train_data_path,args.test_data_path,args.result_path,args.visualization_on_off)



def precompute_train_features(train_datapath,intermediate_layer_modelb4):
    train_image_features = {}

    for filename in os.listdir(train_datapath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(train_datapath, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
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
            train_image_features[filename] = img_features
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

    #b4
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
    

    #b6 
    # model = tf.keras.applications.EfficientNetB6(include_top=False,
    #         weights="imagenet",
    #         input_tensor=None,
    #         input_shape=(224, 224, 3),
    #         pooling=None,)
    # layer_name = 'block6b_add'
    # intermediate_layer_modelb6 = keras.Model(inputs=model.input,
    #                                         outputs=model.get_layer(layer_name).output)

    pretrained_feature = os.path.join(cfg.train_data_path,'train_weights_path')
    data_path = os.path.join(os.path.join(cfg.train_data_path,'image_data'))
    # load or generate features from the train dataset
    if cfg.pretrained_features == 'y':
        train_image_features = torch.load(os.path.join(pretrained_feature, 'train_image_features.pth'), map_location=torch.device('cpu'))
        print("-----------------features loaded-------------------")
    elif cfg.pretrained_features == 'n':
        
        train_image_features = precompute_train_features(data_path,intermediate_layer_modelb4)
        
        if not os.path.isdir(pretrained_feature):
                os.mkdir(pretrained_feature)
        torch.save(train_image_features, os.path.join(pretrained_feature, 'train_image_features.pth'))
        print("-----------------calculate and store freatures-------------------")

    # pull train features from CPU to GPU memory 
    train_image_features_gpu = {}
    for title, feature in train_image_features.items():
        feature_tensor = torch.tensor(feature, dtype=torch.float16).cuda()
        train_image_features_gpu[title] = feature_tensor

    # based on selected similariy method initiate the online training block
    if cfg.anomaly_detection_method =='c':
        percent_similarity = 0.85    
        online_training_int = cosine_training(percent_similarity,train_image_features_gpu)
        print("-----------------Cosine similarity block initiated-------------------")
    elif cfg.anomaly_detection_method =='sf':
        percent_similarity = 0.7
        # train_image_features = torch.load(os.path.join(os.path.join(cfg.train_data_path,'train_weights_path'), 'train_image_features.pth'), map_location=torch.device('cpu'))
        online_training_int = sift_flann_training(data_path,percent_similarity)
        print("-----------------SIFT-FLANN similarity block initiated--------------------")

   
    #variable declaration
    predicted_value = []
    predicted_tag = []
    data_saved = []
    process_time = []
    image_counter = 1 

    #load test image from test datapath
    fig  = plt.figure()

    #infer images in test_data_path
    for filename in os.listdir(cfg.test_data_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            start_time = time.time()
            image_path = os.path.join(cfg.test_data_path, filename)
            test_image = cv2.imread(image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            test_image = cv2.resize(test_image, (224, 224))
            test_image_preprocess = img_to_array(test_image)
            test_image_preprocess = np.expand_dims(test_image_preprocess, axis= 0)
            test_image_preprocess = preprocess_input(test_image_preprocess)

            extracted_feature = intermediate_layer_modelb4.predict(test_image_preprocess)

            if cfg.anomaly_detection_method =='c':
                titles_compared, cal = online_training_int.similarity_block(extracted_feature)
            elif cfg.anomaly_detection_method =='sf':
                titles_compared, cal = online_training_int.similarity_block(image_path)

            data_saved.append(cal)

            if len(titles_compared)>0:
                mu, inv_con_model, threshold = detection_train_block(titles_compared,train_image_features_gpu)
                result_image,anomaly_prediction = detection_test_block(extracted_feature,mu, inv_con_model, threshold,test_image)
                predicted_value.append(anomaly_prediction)

            else:
                print('No similar image found,test image does not match the environment')
                print("Result : Anomaly")
                predicted_value.append(1)
                result_image = test_image

            if cfg.visualization_on_off == 1:
                plt.axis('off')
                plt.imshow(result_image)
                plt.savefig((heatmap_result_path + '/{}').format(filename))
                plt.show(block = False)
                plt.pause(0.5)
                fig.canvas.flush_events()
            else:
                plt.axis('off')
                plt.imshow(result_image)
                plt.savefig((heatmap_result_path + '/{}').format(filename))
            print("Computation time per frame: ", time.time() - start_time, "secs")
            process_time.append(time.time() - start_time)
            image_counter +=1
        plt.close()
    
    #calculating overall results
    avg_data = np.average(data_saved)
    print('Average % data saved from training: ', avg_data,'%')

    # overall processing time
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


    # update the prediction tag
    for n in predicted_value:
        if n==0:
            predicted_tag.append('No Anomaly')
        elif n==1:
            predicted_tag.append('Anomaly')

    # Create a dictionary with the data
    data = {'predicted_values': predicted_value, 'predicted_tag': predicted_tag}

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
   
    # Write the DataFrame to the CSV file
    df.to_csv(result_filepath, index=False)

    print(f"Predictions saved to {result_filepath}")

if __name__ == '__main__':
    configure_gpu()
    cfg = config()
    main(cfg)