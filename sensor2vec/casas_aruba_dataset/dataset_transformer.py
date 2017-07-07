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
OUTPUT_COMPLETE_DATASET = "aruba_complete_dataset.csv"

# Function to check the similarity between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


print "Loading ", INPUT_DATASET
# Load the input dataset in a pd.DataFrame
#idf = pd.read_csv(INPUT_DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
idf = pd.read_csv(INPUT_DATASET, parse_dates=[[0, 1]], header=None, sep=' ')
#idf.columns = ['sensor', 'value', 'activity', 'event']
idf.columns = ["timestamp", 'sensor', 'value', 'activity', 'event']
#idf.index.names = ["timestamp"]

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
#for i in xrange(len(auxdf)):
for i in auxdf.index:
    maxv = 0.0
    candidate = auxdf.loc[i]["value"]
    for value in valid_values:
        sim = similar(auxdf.loc[i]["value"], value)
        if sim > maxv:
            maxv = sim
            candidate = value
    #print auxdf.ix[i]
    #print "   changed to", candidate
    #modify the actual row in idf    
    idf.set_value(i, "value", candidate)    

# In order to verify the change, print idf

index = auxdf.index
print "Weird values modified"
newdf = idf.loc[index]
print newdf.head(10)


# Fill all the activity column with corresponding values

begin_index = idf[idf["event"] == "begin"].index
end_index = idf[idf["event"] == "end"].index

assert(len(begin_index) == len(end_index))

print "Number of activities:", len(begin_index)
for i in xrange(len(begin_index)):
    print i
    activity = idf.loc[begin_index[i]]["activity"]
    idf.loc[begin_index[i]:end_index[i]]["activity"] = activity

# Now replace every NaN in the column "activity" by "None"
idf["activity"].fillna("None", inplace=True)

print "Activities properly set"
print idf.head(60)

# Write the DataFrame to CSV
idf.to_csv(OUTPUT_COMPLETE_DATASET, header=False, index=False)



