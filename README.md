# _Material Classification using Thermal Imaging_
This project was carried out within the scope of Hacettepe University Electrical and Electronics Engineering Undergraduate Program graduation projects. This project was done by me and eyüp enes aytaç in a completely unique way. Here's the repository about our graduation project.

## Table of Contents
- [_Material Classification using Thermal Imaging_](#-material-classification-using-thermal-imaging-)
  * [Abstract](#abstract)
  * [Experiment Setup](#experiment-setup)
  * [Creating a Dataset](#creating-a-dataset)
  * [Obtaining Temperature Readings](#obtaining-temperature-readings)
  * [Feature Extraction](#feature-extraction)
  * [Training Classifiers](#training-classifiers)
  * [Experimental Result](#experimental-result)
  * [Project Video](#project-video)
  * [Used](#used)
  * [Authors](#authors)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'></a></i></small>
## Abstract
The proposed method suggests an alternative approach to the classification step in mainstream recycling processes. It addresses the need for automation and cost-effectiveness in waste management due to the increasing population and waste generation. 
The method involves an active heating unit comprising three thermal lamps and utilizes a FLIR T420 thermal camera. The objects to be classified are heated using the heating unit and then allowed to passively cool down. 
Throughout this process, the thermal camera captures images for observation. These thermal images are subsequently utilized in machine learning algorithms for material classification. 
The primary goal of the proposed method is to automate the classification step in recycling while ensuring affordability and ease of implementation. The accuracy of the system is evaluated based on a dataset collected in the laboratory.

## Experiment Setup
![a](https://github.com/emerttosun/Material-Classification-using-Thermal-Imaging/assets/138903517/5222c153-bf1b-4fc2-bd05-f9f289d880ea)
The setup seen above belongs to one of the first trial setups and we made our first trial and recordings using this setup. The experimental setup for the proposed method includes a FLIR T-400 thermal camera, three OSRAM heater lamps, and a table where the lamps are positioned. 

Various municipal solid waste objects made up of different materials like carton, metal and plastic are used for this work. Special attention was given to select waste items with non-smooth surfaces, ensuring a mixture of both smooth and uneven textures are present in material set. 
## Creating a Dataset
The entire dataset for this project was created from scratch. 

Dataset is include videos which are recorded to heating and cooling processes of objects. Each data consists of approximately 25 seconds of video and each data has a 5 second initial display, a 10 second heating period, and 10 second cooling period. Afterwards temperature over time of each object is used for classification.

## Obtaining Temperature Readings
Thermal videos are collected in grayscale as RGB images do not provide additional information. Grayscale images are easier to store and process, simplifying the processing pipeline and improving algorithmic performance. 
To enhance accuracy, fixed minimum and maximum temperature values are established for the recorded data, with optimal values determined to be 15 and 45 degrees Celsius, respectively.
In the grayscale images, brighter shades of gray indicate hotter temperatures, while darker shades represent colder temperatures, with the darkest shade corresponding to 15 degrees Celsius and the lightest shade corresponding to 45 degrees Celsius.
This standardized temperature range simplifies data processing and ensures consistency across all records, resulting in improved accuracy.

To remove the noise in the video, each frame is filtered with Gaussian Filter. By applying Gaussian filter, the resulting data is rendered more accurate and lower noise, further improving the overall analysis. 
The temperature of the objects is inferred by calculating the average temperature within a Region of Interest (ROI) in the video.  The ROI is selected to cover most of the object, allowing for a more nuanced and accurate understanding of the data.

Here's the example of ROI.

![bitirme1](https://github.com/emerttosun/Material-Classification-using-Thermal-Imaging/assets/138903517/0302f276-cf7f-4b14-b23d-aea2e2992dbd)


After obtaining average value within the ROI in each frame of the video, a low pass filter, Hann filter is applied. The aim is to remove the noise in obtained time series. At the end of this process, a time series that shows pixel intensity over time is obtained. 

## Feature Extraction
Features are extracted from the time series. To extract features from the thermal data, the highest temperature point is identified as the reference point. 
This process involves examining 5-second intervals from both the heating and cooling phases of the data. By analyzing these intervals, relevant features can be derived. 
![Figure_1ddd](https://github.com/emerttosun/Material-Classification-using-Thermal-Imaging/assets/138903517/ea4f40ef-571b-4110-ba50-b7cac5c0a961)

The features are designed to leverage the relationship between a specific point and the preceding points. An exponential decay approach was employed to establish connections between points in the specified range. 
This methodology effectively captures evolving patterns and relationships between the points, resulting in enhanced accuracy for material representation and classification. Our system is causal because it depends only on the present past values.

## Training Classifiers
Prior to training a supervised learning classifier, careful preparation of the input data and corresponding outputs is essential.
This involves feature extraction to enable accurate classification. In our case, the output array is prepared using label encoding, which assigns integer values to each distinct class (metal, plastic, and carton). 
The labels are transformed into a machine-readable format for compatibility with various machine learning algorithms, including Support Vector Machines (SVM).
In our specific case, we have three classes, namely metal, plastic, and carton. For clarity and consistency, these classes are encoded as follows:

- Metal: 0
- Plastic: 1
- Carton: 2


The procedures described above are designed to be applicable to a wide range of machine learning algorithms. 
In addition to these traditional techniques, a deep learning model was trained to compare its effectiveness with other classifiers, requiring only minor modifications to adapt the steps for the different nature of the deep learning approach.

The deep learning model architecture consists of multiple layers of linear transformations and sigmoid activation functions to process input data and generate predictions. The model is adjusted and selected based on tests after several trials.
One-hot encoding is used for the output data to comply with the requirement of the Cross Entropy Loss function utilized in deep learning frameworks. 
The training process involves dividing the data into batches, conducting forward and backward passes to update model parameters, and evaluating performance based on accuracy and loss. 
The best-performing model is saved using the Adam optimizer. 
The appropriate number of epochs is determined by analyzing the relationship between training and testing accuracy to achieve a balance between high accuracy on training data and good generalization on testing data.

## Experimental Result
In the experiment, data was collected, features were extracted, and models were trained using six different classifiers: 
- Support Vector Machine (SVM)
- Decision Tree
- K Nearest Neighbors (KNN) 
- Random Forest 
- Gradient Boosting 
- Deep Learning

The performance of each classifier was evaluated using 30 time series obtained from videos, with 10 time series from each class (metal, plastic, and carton).
None of the time series used in testing were used for training. The results indicate that KNN performed the best, followed by Decision Trees and Artificial Neural Networks (ANN), you can see below. SVM was the least effective classifier.

From the results discussed above, metal and carton classes are easiest to classify, while the hardest one is the plastic. 
This is an expected result since the heat capacity of metal is the highest and the carton is the lowest. Heat capacity of plastic is in between so it is expected to be misclassified.

![image](https://github.com/emerttosun/Material-Classification-using-Thermal-Imaging/assets/138903517/0431537f-7017-4536-906c-94f015a156bb)

## Project Video
You can reach the our project videos here!
https://youtu.be/0mC2CdkLAvA

## Used
- Python
- Scikit-learn
- Pytorch
- OpenCV
- Pandas
- PyQt

## Authors
- Erkani Mert Tosun - [emerttosun](https://github.com/emerttosun)
- Eyüp Enes Aytaç - [EyupAytac](https://github.com/EyupAytac)



