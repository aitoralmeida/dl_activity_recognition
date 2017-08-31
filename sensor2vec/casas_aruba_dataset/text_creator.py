# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:05:02 2017

@author: gazkune
"""
import sys
import pandas as pd
import numpy as np

# The input dataset 
DATASET = "aruba_complete_dataset.csv"
# Directory for output text files
OUTPUT_DIR = 'action2vec'

# All the actions in the same row; all the sensors; numeric values appended to sensor name
CONTINUOUS_COMPLETE_NUMERIC = OUTPUT_DIR + 'continuous_complete_numeric.txt'
# All the actions in the same row; all the sensors; numeric values appended to sensor name in ranges (2)
CONTINUOUS_COMPLETE_RANGES = OUTPUT_DIR + 'continuous_complete_ranges_2.txt'
# All the actions in the same row; no temperature sensors
CONTINUOUS_NO_T = OUTPUT_DIR + 'continuous_no_t.txt'

# Every row is an activity; all the sensors; numeric values appended to sensor name
LINED_COMPLETE_NUMERIC = OUTPUT_DIR + 'lined_complete_numeric.txt'
# Every row is an activity; all the sensors; numeric values appended to sensor name in ranges (2)
LINED_COMPLETE_RANGES = OUTPUT_DIR + 'lined_complete_ranges_2.txt'
# Every row is an activity; no temperature sensors
LINED_NO_T = OUTPUT_DIR + 'lined_no_t.txt'

# --------------------------------------------------------
# CONTINUOUS_COMPLETE_NUMERIC case    
def storeContinuousCompleteNumeric(idf):
    sensors = idf["sensor"].values
    values = idf["value"].values

    try:
        assert(len(sensors) == len(values))
    except AssertionError:
        print 'Number of sensors and values are not equal; sensors:', len(sensors), 'values:', len(values)

    words = sensors + '_' + values
    np.savetxt(CONTINUOUS_COMPLETE_NUMERIC, words, fmt='%s', newline=" ")    

# --------------------------------------------------------
# CONTINUOUS_NO_T case
def storeContinuousNoT(idf):
    auxdf = idf[np.logical_not(idf["sensor"].str.contains("T0"))]
    sensors = auxdf["sensor"].values
    values = auxdf["value"].values

    try:
        assert(len(sensors) == len(values))
    except AssertionError:
        print 'Number of sensors and values are not equal; sensors:', len(sensors), 'values:', len(values)

    words = sensors + '_' + values
    np.savetxt(CONTINUOUS_NO_T, words, fmt='%s', newline=" ")

# --------------------------------------------------------
# LINED_COMPLETE_NUMERIC
def storeLinedCompleteNumeric(idf):
    previous_activity = idf.loc[0]['activity']
    with open(LINED_COMPLETE_NUMERIC, "w") as text_file:
        for i in idf.index:
            print i,
            if idf.loc[i]['activity'] != previous_activity:
                previous_activity = idf.loc[i]['activity']
                #print("\n", file=text_file)
                text_file.write("\n")
            
            word = idf.loc[i]['sensor'] + '_' + idf.loc[i]['value']
            #print("{} ".format(word), file=text_file)
            text_file.write(word + ' ')

# --------------------------------------------------------
# LINED_NO_T
def storeLinedNoT(idf):
    auxdf = idf[np.logical_not(idf["sensor"].str.contains("T0"))]
    previous_activity = auxdf.loc[0]['activity']
    with open(LINED_NO_T, "w") as text_file:
        for i in auxdf.index:
            print i,
            if auxdf.loc[i]['activity'] != previous_activity:
                previous_activity = auxdf.loc[i]['activity']
                #print("\n", file=text_file)
                text_file.write("\n")
            
            word = auxdf.loc[i]['sensor'] + '_' + auxdf.loc[i]['value']
            #print("{} ".format(word), file=text_file)
            text_file.write(word + ' ')


# --------------------------------------------------------
# CONTINUOUS_COMPLETE_RANGES
def storeContinuousCompleteRanges(idf):
    tempdf = idf[idf["sensor"].str.contains("T0")]
    mintemp = round(float(min(tempdf["value"])))
    maxtemp = round(float(max(tempdf["value"])))
    increment = 2 # test with two degrees of increment in temperature

    bins = np.arange(mintemp, maxtemp, increment)
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

    words = sensors + '_' + values
    np.savetxt(CONTINUOUS_COMPLETE_RANGES, words, fmt='%s', newline=" ")
    
# --------------------------------------------------------
# LINED_COMPLETE_RANGES
def storeLinedCompleteRanges(idf):
    tempdf = idf[idf["sensor"].str.contains("T0")]
    mintemp = round(float(min(tempdf["value"])))
    maxtemp = round(float(max(tempdf["value"])))
    increment = 2 # test with two degrees of increment in temperature

    bins = np.arange(mintemp, maxtemp, increment)
    temperatures = tempdf["value"].values.astype(float)
    # Discretize temperatures
    inds = np.digitize(temperatures, bins)
    temperatures = bins[inds-1]

    tempdf["value"] = temperatures.astype(str)
    idf.ix[tempdf.index, "value"] = tempdf["value"]
    
    previous_activity = idf.loc[0]['activity']
    with open(LINED_COMPLETE_RANGES, "w") as text_file:
        for i in idf.index:
            print i,
            if idf.loc[i]['activity'] != previous_activity:
                previous_activity = idf.loc[i]['activity']
                #print("\n", file=text_file)
                text_file.write("\n")
            
            word = idf.loc[i]['sensor'] + '_' + idf.loc[i]['value']
            #print("{} ".format(word), file=text_file)
            text_file.write(word + ' ')
    

# --------------------------------------------------------
# Main function
def main(argv):
    # Load Aruba dataset
    idf = pd.read_csv(DATASET, parse_dates=[0], header=None, sep=',')
    idf.columns = ["timestamp", 'sensor', 'value', 'activity', 'event']
    #storeContinuousCompleteNumeric(idf)
    #storeContinuousNoT(idf)
    #storeLinedCompleteNumeric(idf)
    #storeLinedNoT(idf)    
    #storeContinuousCompleteRanges(idf)    
    storeLinedCompleteRanges(idf)

if __name__ == "__main__":
   main(sys.argv)

