import os
import csv
samples=[]
pos=0
zer=0
neg=0
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(line[0]!="center"):
           if(float(line[3])>0):
              pos=pos+1
           if(float(line[3])==0):
              zer=zer+1
           if(float(line[3])<0):
              neg=neg+1
           samples.append(float(line[3]))

print(pos,zer,neg)
print(len(samples))
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
samples=np.array(samples)
plt.hist(samples, bins=30, alpha=0.5)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


