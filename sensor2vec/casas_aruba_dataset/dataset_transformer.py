# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:28:26 2017

@author: gazkune
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher 

# Input data
INPUT_DATASET = "data"
# Dataset formatted in our own format
OUTPUT_DATASET = "aruba_dataset.csv"

# Function to check the similarity between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Load the input dataset in a pd.DataFrame
idf = pd.read_csv(INPUT_DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
idf.columns = ['sensor', 'value', 'activity', 'event']
idf.index.names = ["timestamp"]

#print idf.head(50)

#print "------------------------------------"

#print idf[idf.sensor.isnull()]

# Store in tempdf only temperature sensor information only for data 
# visualizations purposes

"""
tempdf = idf[idf["sensor"].str.contains("T0")]
temperatures = np.array(tempdf["value"].values)
temperatures = temperatures.astype(np.float)

avg = np.mean(temperatures)
median = np.median(temperatures)
std = np.std(temperatures)
maxtemp = np.max(temperatures)
mintemp = np.min(temperatures)

print "Temperatures (mean, median, std, max, min):", avg, median, std, maxtemp, mintemp
temp_thr = 35.0
above_thr = np.where(temperatures > temp_thr)[0].size
print "Temperatures above", temp_thr, above_thr, float(above_thr)/float(temperatures.size)*100.0, "%"
#print idf[idf["value"] == str(int(maxtemp))]

plt.hist(temperatures)
plt.title("Temperature distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

fig = plt.gcf()
"""

# Clean weird sensor values (OFFcc, ONcc, etc)
# We assume that valid values are: [ON, OFF, CLOSE, OPEN]
valid_values = ["ON", "OFF", "CLOSE", "OPEN"]
# store in auxdf only motion and door sensors
auxdf = idf[np.logical_or(idf["sensor"].str.contains("M"), idf["sensor"].str.contains("D"))]
#print auxdf.head(10)

auxdf = auxdf[auxdf["value"] != "ON"]
auxdf = auxdf[auxdf["value"] != "OFF"]
auxdf = auxdf[auxdf["value"] != "CLOSE"]
auxdf = auxdf[auxdf["value"] != "OPEN"]

print auxdf.head(10)

# At this point, auxdf has only invalid sensor values
for i in xrange(len(auxdf)):
    maxv = 0.0
    candidate = auxdf.ix[i]["value"]
    for value in valid_values:
        sim = similar(auxdf.ix[i]["value"], value)
        if sim > maxv:
            maxv = sim
            candidate = value
    #print auxdf.ix[i]
    #print "   changed to", candidate
    #modify the actual row in idf
    index = auxdf.index[i]
    idf.loc[index]["value"] = candidate
    #print idf.loc[index]

# In order to verify the change, print idf
index = auxdf.index
print "Weird values modified"
newdf = idf.loc[index]
print newdf.head(10)



"""
# Code to identify weird sensor names

auxdf = idf[idf["sensor"].str.contains('M0')==False]
auxdf = auxdf[auxdf["sensor"].str.contains('D0')==False]
auxdf = auxdf[auxdf["sensor"].str.contains('T0')==False]

print auxdf.head(10)
"""


