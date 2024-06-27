# Sift & Flann method training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
import glob
import cv2

# enabling gpu
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class sift_flann_training():
    def __init__(self,train_datapath,percent_similarity):
        self.train_datapath = train_datapath
        self.percent_similarity = percent_similarity
    
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

        for filename in os.listdir(self.train_datapath):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.train_datapath, filename)
                image = cv2.imread(image_path)
                titles.append(filename)
                train_data_compare.append(image)

        # print('train image set', len(train_data_compare))
        #feature matching with FLANN
        titles_compared = []
        for image_to_compare, title in zip(train_data_compare, titles):
            # 1) Check if 2 images are equals 
            # print('here')
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
            percentage_similarity = percentage_similarity/10
            # print(percentage_similarity)
            if percentage_similarity>self.percent_similarity:           #texture = 40 and for solid 65/70 percentage
                titles_compared.append(title)

            
        n = len(train_data_compare) - len(titles_compared)
        d = len(train_data_compare)
        cal = n / d * 100
        data_saved.append(cal)
        print(cal, "% data saved from unwanted training")
        return titles_compared, cal
    
