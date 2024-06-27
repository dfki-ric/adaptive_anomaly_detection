#visualization 
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def config():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-t','--test_data_path', type=str, help="Folder containing images to tested", required=True)

    parser.add_argument('-d','--true_value_path', type=str, help="Folder containing true value csv for test data", required=True)

    parser.add_argument('-r','--results_path', type=str, help="Folder containing results(generated images and predictions.csv)", required=True)

    parser.print_help()
    # Execute parse_args()
    args = parser.parse_args()

    return args

def main(cfg):

    # Load your CSV file with true and predicted values into a DataFrame
    df = pd.read_csv(os.path.join(cfg.results_path,'prediction.csv'))
    df_v = pd.read_csv(os.path.join(cfg.true_value_path,'true_value.csv'))

    #accuracy calculation
    true_val = df_v['true_values']
    predicted = df['predicted_values']
    print("Accuracy :",
          accuracy_score(true_val, predicted))
    print("Confusion matrix", confusion_matrix(true_val, predicted))
    cm = confusion_matrix(true_val, predicted)
    df = pd.DataFrame(cm).transpose()
    df.to_csv(os.path.join(cfg.results_path, 'confusion_matrix.csv'))
    report = classification_report(true_val, predicted, output_dict=True)
    CR = pd.DataFrame(report).transpose()
    CR.to_csv(os.path.join(cfg.results_path, 'classification_report.csv'))
    print(CR)

    # Initialize an empty list to store the index (x-axis values)
    index_data = []

    # Initialize empty lists to store true and predicted values
    true_values = []
    predicted_values = []

    # Define the path to the folder containing images
    result_folder = os.path.join(cfg.results_path,"heatmap")
    test_datapath = cfg.test_data_path

    # Get a list of image files in the folder
    image_files = sorted([os.path.join(test_datapath, img) for img in os.listdir(test_datapath) if img.endswith(('.jpg', '.png'))])

    # Create a figure and subplots for the live plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    plt.xticks(rotation=45)

    # Function to update the live plot
    def update(frame):
        # Append the next index (x-axis value)
        index_data.append(len(index_data))

        # Append the next true and predicted values
        true_values.append(true_val.iloc[len(index_data) - 1])
        predicted_values.append(predicted.iloc[len(index_data) - 1])

        # Display the original image in the first subplot
        img_path = image_files[len(index_data) - 1]
        original_img = cv2.imread(img_path)
        ax1.clear()
        ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        ax1.axis('off')

        # Load the detected image using OpenCV
        detected_img_path = os.path.join(result_folder, os.path.basename(img_path))
        detected_img = cv2.imread(detected_img_path)
        ax2.clear()
        ax2.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        ax2.axis('off')
        ax2.set_aspect('auto')  # Set aspect ratio to be equal to ax1

        # Plot the true values in blue and predicted values in red
        ax3.clear()
        ax3.plot(index_data, true_values, label='True', color='blue')
        ax3.plot(index_data, predicted_values, label='Predicted', color='green')
        ax3.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('true and predicted values')
        ax3.legend()

    # Create an animation to update the plot every n milliseconds
    ani = FuncAnimation(fig, update, interval=1000)  # Update every 1000 milliseconds (1 second)

    # Show the live plot
    plt.show()
if __name__ == '__main__':
    cfg = config()
    main(cfg)
