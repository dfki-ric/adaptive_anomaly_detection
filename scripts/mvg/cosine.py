#Cosine similarity method training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch



class cosine_training():
    def __init__(self,percent_similarity,train_image_features):
        
        self.percent_similarity = percent_similarity
        # Initialize a dictionary to store titles and computed features for training images
        self.train_image_features = train_image_features

    # finds similarity between test image and training dataset using feature comparison with cosine method

    def similarity_block(self,extracted_feature):
        #self.test_image = test_image
        
        feature_vector = extracted_feature
        a, b, c, n = feature_vector.shape
        feature_vector = feature_vector.reshape(b*c,n)

        titles_compared = [] 
        new_dict = {}

        test_feature_vector = torch.tensor(feature_vector, dtype=torch.float32).cuda()
    
        for train_img_title, train_features in self.train_image_features.items():
 
            a, b, c, n = self.train_image_features[train_img_title].shape
            train_feature_vector= self.train_image_features[train_img_title].reshape(b*c,n)
            image_similarity_cosine = torch.nn.functional.cosine_similarity(test_feature_vector,
                                                        train_feature_vector).cuda()

            if torch.max(image_similarity_cosine) >= self.percent_similarity:
                titles_compared.append(train_img_title)
                new_dict[train_img_title] = self.train_image_features[train_img_title]

        
        n = len(self.train_image_features) - len(titles_compared)
        d = len(self.train_image_features)
        cal = n / d * 100
        print(cal, "% data saved from unwanted training")
        return titles_compared , cal
