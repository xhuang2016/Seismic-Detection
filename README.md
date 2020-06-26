[![](https://img.shields.io/badge/license-GPL--3.0-blue)](https://www.gnu.org/licenses/)
[![](https://img.shields.io/badge/Python-3.7.2-green)](https://www.python.org/downloads/release/python-372/)

# CrowdQuake: A Networked System of Low-Cost Sensors for Earthquake Detection via Deep Learning

<!---This is a Python3 implementation of Convolutional Recurrent Neural Networks for the task of binary classification of seismic detection, as described in our paper.--> 

## Overview

Here we provide the implementation of Artificial Neural Network (ANN) and Convolutional Recurrent Neural Network (CRNN) used in our paper that appears in ACM KDD 2020. [![](https://img.shields.io/badge/Source-Download-yellow)](https://drive.google.com/file/d/1QA0Ua8mF9_aW_9v_xpgmmzTA0gWF5oZL/view?usp=sharing)  
The repository is organised as follows:
* ```requirements.txt```: All required Python libraries to run the code.
* ```Code/Processing.py```: Source code for processing the earthquake data.
* ```Code/ANN.py```: Implementation of ANN.
* ```Code/CRNN.py```: Implementation of CRNN.
* ```Code/ANN_kfold.py```: K-fold cross validation for ANN.
* ```Code/CRNN_kfold.py```: K-fold cross validation for CRNN.
* ```Code/PerformanceMetrics.py```: Calculate the Performance Metrics.


## Requirements
<!---numpy==1.16.1--> 
<!---pandas=0.24.1--> 
<!---obspy==1.1.1--> 
<!---scikit-learn==0.21.2--> 
<!---tensorflow-gpu==1.14--> 
<!---Keras==2.2.4--> 
<!---matplotlib==3.0.3--> 

[![](https://img.shields.io/badge/numpy-1.16.1-green)](https://numpy.org/devdocs/index.html)
[![](https://img.shields.io/badge/pandas-0.24.1-green)](https://pandas.pydata.org/pandas-docs/stable/index.html)
[![](https://img.shields.io/badge/obspy-1.1.1-green)](https://docs.obspy.org/)
[![](https://img.shields.io/badge/scikit--learn-0.21.2-green)](https://scikit-learn.org/stable/index.html)
[![](https://img.shields.io/badge/tensorflow-gpu--1.14-green)](https://www.tensorflow.org/)
[![](https://img.shields.io/badge/Keras-2.2.4-green)](https://keras.io/)
[![](https://img.shields.io/badge/matplotlib-3.0.3-green)](https://matplotlib.org/3.0.3/index.html)

```bash
$ pip install -r requirements.txt
```

In addition, CUDA 10.0 and cuDNN 7.4 have been used.


## Dataset
The dataset contains two classes of three-component time-series acceleration waveforms:
1. Earthquake: Download from National Research Institute for Earth Science and Disaster Resilience (NIED).
  [![](https://img.shields.io/badge/Earthquake-Download-yellow)](http://www.kyoshin.bosai.go.jp/kyoshin/data/index_en.html)

> Note: In our paper, we use two earthquake datasets. 
> 1. 2299 K-NET records of Japan earthquakes from Jan. 1st 1996 to May 31th 2019, each of whose x-axis component peak ground acceleration (PGA) is greater than 0.1 gravity (g). 
> 2. 8973 K-NET records of Japan earthquakes from Jan. 1st 1996 to May 31th 2019, each of whose x-axis component PGA is greater than 0.05 g.

> To download the earthquake data for reproducing our results, please: 
>> + Visit [NIED](http://www.kyoshin.bosai.go.jp/kyoshin/data/index_en.html) (registration is required to download data).  
>> + Select `Download` -- `Data Download after Search for Data`.  
>> + Set `Network` -- `K-NET`.  
>> + Set `Peak acceleration` -- `from 100 to 5000` (i.e., > 0.1 g) or `from 50 to 5000` (i.e., > 0.05 g).  
>> + Set `Recording start time`.  
>> + Click `Submit`, then select all records in `Data List`.  
>> + Click `Download All Channels Data`.  
>> + Notice that, due to the constraint of the website, only headmost 1200 data will be displayed in `Data List`. Please change the range of `Recording start time`, then repeat above procedures to download all data in batches.

2. Non-Earthquake: Background noise of the low-cost sensors measured in several environments and various human activities recorded by our low-cost sensors.
  [![](https://img.shields.io/badge/Non--Earthquake-Download-yellow)](https://drive.google.com/file/d/11sivVlx7z-cBwjBWPNY9D2Wmfv-FY-CM/view?usp=sharing)

You can also use your own dataset, but the dataset should contain three-component time-series acceleration signals.

## Models

You can choose between the following models: 
* ANN: A multi-layer perceptron with three input features (IQR,ZC,and CAV) and a hidden layer of 5 neurons. See more details in our paper or [MyShake](https://advances.sciencemag.org/content/2/2/e1501055) paper.
<!---* CNN--> 
* CRNN: A combination of Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN). See more details in our paper.

## Running the code
1. Download the data as described above.
2. Run ```Code/Processing.py``` to process the raw data.
3. Run ```Code/ANN.py``` or ```Codes/CRNN.py``` to train and test each model.
4. Run ```Models/ANN_kfold.py``` or ```Models/CRNN_kfold.py``` for the k-fold cross validation for each model.
5. Run ```Code/PerformanceMetrics.py``` to calculate the performance metrics.
> Note:
> 1. To reproduce our results, please download the same set of data as we used in the paper.  
> 2. For different test cases, please change the settings as described in each code file.

<!---## Cite--> 

<!---Please cite our paper if you use this code in your own work:--> 
