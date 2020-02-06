#!/usr/bin/env python
"""
====================================
Processing Earthquake data
====================================
"""
import pandas as pd
import numpy as np
import obspy
from obspy import read
import os
import glob

## Download Eearthquake (EQ) data from NIED (http://www.kyoshin.bosai.go.jp/kyoshin/data/index_en.html)
## Uncompress the downloaded **.tar data file to the folder 'raw'
## Create the 'EQ' folder to save the processed data

# Loction of raw EQ data
file_dir = 'raw'
# Loction to save the processed EQ data
write_path = 'EQ/'

for root, dirs, files in os.walk(file_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.EW':
            ew = read(os.path.join(root, file))
        if os.path.splitext(file)[1] == '.NS':
            ns = read(os.path.join(root, file))
        if os.path.splitext(file)[1] == '.UD':
            ud = read(os.path.join(root, file))
            tr_ew = ew[0]
            tr_ns = ns[0]
            tr_ud = ud[0]

            # Convert [gal] to [g]
            data_ew = tr_ew.data*0.00101972*ew[0].stats.calib*100
            data_ns = tr_ns.data*0.00101972*ns[0].stats.calib*100
            data_ud = tr_ud.data*0.00101972*ud[0].stats.calib*100

            # Move to ground by subtracting the mean
            data_ew = data_ew - np.mean(data_ew)
            data_ns = data_ns - np.mean(data_ns)
            data_ud = data_ud - np.mean(data_ud)

            filename = os.path.splitext(file)[0]

            # Select the EQ records whose x-axis component PGA is greater than 0.1/0.05 [g]
            if np.max(data_ew) > 0.1:
            # if np.max(data_ew) > 0.05:
                np.savetxt(write_path + filename + '.txt', np.c_[data_ew,data_ns,data_ud])
