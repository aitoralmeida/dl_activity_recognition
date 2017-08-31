# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:27:31 2017

@author: gazkune

Script to generate several csv files for aruba dataset
"""


import sys
import pandas as pd
import numpy as np

# The input dataset 
DATASET = "aruba_complete_dataset.csv"

# Output datasets
COMPLETE_NUMERIC = "aruba_complete_numeric.csv"
NO_T = "aruba_no_t.csv"
INCR = 2 # define the temperature range to be used
COMPLETE_RANGES = "aruba_complete_ranges_" + str(INCR) + ".csv"

# --------------------------------------------------------
# COMPLETE_NUMERIC case    
def storeCompleteNumeric(idf):
    sensors = idf["sensor"].values
    values = idf["value"].values

    try:
        assert(len(sensors) == len(values))
    except AssertionError:
        print 'Number of sensors and values are not equal; sensors:', len(sensors), 'values:', len(values)

    actions = sensors + '_' + values
    idf["action"] = actions
    # Decision: I will not store 'sensor', 'value' and 'event' columns    
    idf.to_csv(COMPLETE_NUMERIC, columns=['timestamp', 'action', 'activity'], header=False, index=False)
    

# --------------------------------------------------------
# NO_T case
def storeNoT(idf):
    auxdf = idf[np.logical_not(idf["sensor"].str.contains("T0"))]
    sensors = auxdf["sensor"].values
    values = auxdf["value"].values

    try:
        assert(len(sensors) == len(values))
    except AssertionError:
        print 'Number of sensors and values are not equal; sensors:', len(sensors), 'values:', len(values)

    actions = sensors + '_' + values
    auxdf["action"] = actions
    # Decision: I will not store 'sensor', 'value' and 'event' columns    
    auxdf.to_csv(NO_T, columns=['timestamp', 'action', 'activity'], header=False, index=False)

# --------------------------------------------------------
# COMPLETE_RANGES
def storeCompleteRanges(idf):
    tempdf = idf[idf["sensor"].str.contains("T0")]
    mintemp = round(float(min(tempdf["value"])))
    maxtemp = round(float(max(tempdf["value"])))    

    bins = np.arange(mintemp, maxtemp, INCR)
    temperatures = tempdf["value"].values.astype(float)
    # Discretize temperatures
    inds = np.digitize(temperatures, bins)
    temperatures = bins[inds-1]

    tempdf["value"] = temperatures.astype(str)
    idf.ix[tempdf.index, "value"] = tempdf["value"]

    sensors = idf["sensor"].values
    values = idf["value"].values

    try:
        assert(len(sensors) == len(values))
    except AssertionError:
        print 'Number of sensors and values are not equal; sensors:', len(sensors), 'values:', len(values)

    actions = sensors + '_' + values
    idf["action"] = actions
    # Decision: I will not store 'sensor', 'value' and 'event' columns    
    idf.to_csv(COMPLETE_RANGES, columns=['timestamp', 'action', 'activity'], header=False, index=False)


# Load Aruba dataset
idf = pd.read_csv(DATASET, parse_dates=[0], header=None, sep=',')
idf.columns = ["timestamp", 'sensor', 'value', 'activity', 'event']

# Generate the desired output
storeCompleteNumeric(idf)
storeNoT(idf)
storeCompleteRanges(idf)