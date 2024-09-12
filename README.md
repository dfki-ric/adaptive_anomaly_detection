# Anomaly Detection

## Overview

Welcome to the Anomaly Detection Research Project repository. This project aims to advance the field of anomaly detection in technical applications using innovative techniques, including transfer learning, SIFT-FLANN, and cosine similarity. The project's primary goal is to develop a dynamic and responsive framework for online adaptive anomaly detection.


## Background

The field of anomaly detection is crucial for various technical applications, from manufacturing quality control to real-time surveillance. Traditional methods often rely on rule-based thresholds or supervised learning, which can be impractical due to the need for labeled data. This project explores unsupervised approaches, such as transfer learning, SIFT-FLANN, and cosine similarity, to overcome these limitations and enhance anomaly detection.

## Research Goals

- Develop an online adaptive anomaly detection framework.
- Leverage transfer learning principles to enhance adaptability.
- Integrate SIFT-FLANN and cosine methods with transfer learning.
- Use of MVG method as normality model.
- Compare the proposed framework with the state of the art.
- Explore the robustness and adaptability of the framework in various technical environments.

## Repository Structure

The repository is organized as follows:

- `Cosine.py and SIFT_FLANN.py`: Contains the source code for the anomaly detection methods.
- `NN.py`: Includes Neural networks used in experiments.
- `Main.py`: This is the main execution file to be executed.
- `visualization.py`: This file generates results after the algorithm is executed completely.
- `requirements.txt`: Contains supporting libraries for creating virtual environment.

## Execution

To get started with this project, follow these steps:

1. Clone the repository to your local machine.

2. Create a virtual environment using `requirements.txt`

3. Datasets: The datasets used for the experimetns can be downloaded from the sources mentioned below:
   - [Mockup](https://doi.org/10.5281/zenodo.8319589)
   - [ISS Panel](https://doi.org/10.5281/zenodo.8321662)
   - [MvTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)
   - [VisA](https://paperswithcode.com/dataset/visa)

4. From the virtual environment execute the `Main.py` with following arguments:
   - `-m, --anomaly_detection_method` - Select either `C` for Cosine or `SF` for SIFT-FLANN method.
   - `-f, --pretrained_features`    - Select either `y` to load pretrained features or `n` to compute train image dataset features
   - `-d, --train_data_path` - Provide the path containing images to be trained (/path/to/images/'.jpg', '.png', '.JPG')
   - `-t, --test_data_path` - Provide the path containing images to be tested for anoamly detection (/path/to/images/'.jpg', '.png', '.JPG')
   - `-r, --result_path` - Select the directory to save the generated results by proving the path to it.
   - (optional) `-v, --visualize_detection_on_off` - Choose either `0` == 'on' and `1` == 'off' to visualize the detections while execution of the algorithm.

   ```shell
   (venv)$python3 Main.py -m C -d /path/to/images/*.jpg -t /path/to/images/*.jpg -r /directory/to/save/the/results -v 0
   
(optional)

5. To visualize the generated results with classification report and compare true vs predictions made, run the `visualization.py` with  following arguments:
- `-t, --test_data_path` - Provide the path containing images to be tested for anoamly detection (/path/to/images/'.jpg', '.png', '.JPG'
- `-d, --true_value_path` - Provide the path to .csv where true values are stored for the test data.
- `-r, --results_path` - Provide the path for the folder containing generated results.

```shell
   (venv)$python3 visualization.py -t /path/to/images/*.jpg -d /path/to/true/values/.csv -r /path/conatining/folder/of/results.
```

## Testing

To check the correctness of the code, following conditions should be met:
- The executes without any errors.
- The `Main.py` when executed with correct arguments should display the following results:
   1. Result : Anomaly/ No-Anomaly
   2. Computation time per frame:
   3. Average % data saved from training for every frame:
   4. Average computation time per frame:

This also saves the results generated in the `results` folder. This folder will contain `masked image` results and predicted values in a csv.

- (optional) `visualization.py` when executed with correct arguments should display the following results:
   1. Classification report
   2. A window opens displaying the true image, detected image and the live graph of true vs predicted values.

## License
The code is distributed under the [3-Clause BSD license](LICENSE)

## Citation
(tbd)

### Maintainer / Authors / Contributers

- Siddhant Shete
"Online_Adaptive_Anomaly_Detection_for_Defect_Identification_in_Aircraft_Assembly" was initiated in SeMoSys project and is currently developed at the
Robotics Innovation Center of the German Research Center for ArtificialIntelligence (DFKI) in Bremen.

"Online_Adaptive_Anomaly_Detection_for_Defect_Identification_in_Aircraft_Assembly"  has been funded by German Federal Ministry of Economic Affairs and
Climate Action (BMWK, grant number 20W1922F).

[//]: <> (add logos of funding agencies / DFKI / University here)


Copyright 2023, Siddhant Shete, DFKI RIC
