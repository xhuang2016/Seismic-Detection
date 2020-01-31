![](https://img.shields.io/badge/license-GPL--3.0-blue)
[![](https://img.shields.io/badge/Python-3.7.2-green)](https://www.python.org/downloads/release/python-372/)

# Seismic Detection

This is a Python3 implementation of Convolutional Recurrent Neural Networks for the task of binary classification of seismic detection, as described in our paper:



## Requirements
<!---*tensorflow (1.14)--> 
<!---* Keras (2.2.4)--> 
<!---* scikit-learn (0.21.2)--> 
<!---* pandas (0.24.1)--> 
<!---* numpy (1.16.1)--> 
<!---* obspy (1.1.1)--> 
<!---* matplotlib (3.0.3)--> 


[![](https://img.shields.io/badge/tensorflow-1.14-green)](https://www.tensorflow.org/)
[![](https://img.shields.io/badge/Keras-2.2.4-green)](https://keras.io/)
[![](https://img.shields.io/badge/scikit--learn-0.21.2-green)](https://scikit-learn.org/stable/index.html)
[![](https://img.shields.io/badge/pandas-0.24.1-green)](https://pandas.pydata.org/pandas-docs/stable/index.html)
[![](https://img.shields.io/badge/numpy-1.16.1-green)](https://numpy.org/devdocs/index.html)
[![](https://img.shields.io/badge/obspy-1.1.1-green)](https://docs.obspy.org/)
[![](https://img.shields.io/badge/matplotlib-3.0.3-green)](https://matplotlib.org/3.0.3/index.html)

```bash
$ pip install -r requirements.txt
```


## Data
The dataset contains two classes of three-component time-series acceleration waveforms:
1. Earthquake: Downloaded from National Research Institute for Earth Science and Disaster Resilience (NIED).
    [![](https://img.shields.io/badge/downloads-Earthquake-yellow)](http://www.kyoshin.bosai.go.jp/kyoshin/data/index_en.html)

> Note: We use a total of 2299 K-NET records of Japan earthquakes from Jan. 1st 1996 to May 31th 2019, each of whose x-axis component peak ground acceleration (PGA) is greater than 0.1 gravity (g). To reproduce the results, please sign in NIED, then 

2. Non-Earthquake: Background noise of the low-cost sensors measured in several environments and various human activities recorded by our low-cost sensors.
    ![](https://img.shields.io/badge/downloads-Non--Earthquake-yellow)

You can also use your own data, but the data should contain three-component time-series accelerations.

## Models

You can choose between the following models: 
* ANN
<!---* CNN--> 
* CRNN

## Running the code


## Cite

Please cite our paper if you use this code in your own work:

```

```
