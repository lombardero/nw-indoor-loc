import csv

# Selected AP IDs (11th floor):
ap1 = '70:df:2f:52:83:6a'
ap2 = '70:df:2f:44:43:cd'
ap3 = '70:df:2f:44:d6:e1'
ap4 = '70:df:2f:44:d6:eb'

train = {}
test = {}

coord_train = {}
coord_test = {}

rowidx = 0
count = ['dummy',0,0,0,0]
row = [-91,-91,-91,-91]

def additem(apidx, trtest):
    global data
    global count
    global rowidx
    global row
    global train
    global test
    if count[apidx] == 0:
        row[apidx-1] = int(data[2])
    else:
        if trtest == 'train':
            train[rowidx] = row
        else:
            test[rowidx] = row
        rowidx += 1
        row = [None,None,None,None]
        row[apidx-1] = int(data[2])
        count = ['dummy',0,0,0,0]
    count[apidx] += 1
    # If firts AP reading for point, add it

# Collecting the AP signals from the train data
with open('11train.txt') as datxt:
    rowidx = 1
    count = ['dummy',0,0,0,0]
    row = [-91,-91,-91,-91]
    for line in datxt:
        line = line.split(' ')
        data = []
        for char in line:
            if char != '':
                data.append(char)
        if len(data) > 4:
            if data[1] == ap1:
                additem(1, 'train')
            if data[1] == ap2:
                additem(2, 'train')
            if data[1] == ap3:
                additem(3, 'train')
            if data[1] == ap4:
                additem(4, 'train')

# Collecting the AP signals from the test data
with open('11test.txt') as datxt:
    rowidx = 1
    count = ['dummy',0,0,0,0]
    row = [-91,-91,-91,-91]
    for line in datxt:
        line = line.split(' ')
        data = []
        for char in line:
            if char != '':
                data.append(char)
        if len(data) > 4:
            if data[1] == ap1:
                additem(1, 'test')
            if data[1] == ap2:
                additem(2, 'test')
            if data[1] == ap3:
                additem(3, 'test')
            if data[1] == ap4:
                additem(4, 'test')

# Getting the x-y coordinates for the train data
with open('11readings_train.csv') as csvFile:
    readCSV = csv.reader(csvFile, delimiter=';')
    for readrow in readCSV:
        coord_train[int(readrow[0])] = [float(readrow[1]),float(readrow[2])]

# Getting the x-y coordinates for the test data
with open('11readings_test.csv') as csvFile:
    readCSV = csv.reader(csvFile, delimiter=';')
    for readrow in readCSV:
        coord_test[int(readrow[0])] = [float(readrow[1]),float(readrow[2])]

# 1 - NEAREST NEIGHBOUR
# 1.1 - Predicting using Nearest Neighbour
from scipy.spatial import distance

predictionn = {}
error = 1e10
loss = 0
point = 0

for test_point in test:
    for train_point in train:
        dst = distance.euclidean(test[test_point], train[train_point])
        if dst < error:
            error = dst
            point = train_point
    predictionn[test_point] = coord_train[point]

print(predictionn)
dist_nn = []
total_loss = 0
# 1.2 - Computing Loss
for key in predictionn:
    dist = distance.euclidean(coord_test[key], predictionn[key])
    dist_nn.append(dist)
    total_loss += dist


print(dist_nn)
print(total_loss/20)
        
# 2 - 3-NEAREST NEIGHBOUR
# 2.1 - Predicting using the Three Nearest Neighbours
import statistics

prediction = {}
loss = 0
point = 0
distances = []

for test_point in test:
    # print('Test point:',test_point)
    for train_point in train:
        # print('Train point:',train_point)
        distances.append([train_point,distance.euclidean(test[test_point], train[train_point])])
        # print('APs test, train:', test[test_point], train[train_point])
        # print('Train point, dist:',[train_point,distance.euclidean(test[test_point], train[train_point])])
    distances.sort(key=lambda x: x[1])
    distances = distances[0:3]
    x_list = []
    y_list = []
    for dist in distances:
        x_list.append(coord_train[dist[0]][0])
        x_pred = statistics.mean(x_list)
        y_list.append(coord_train[dist[0]][1])
        y_pred = statistics.mean(y_list)

    prediction[test_point] = [x_pred, y_pred]
    distances = []


dist_3nn = []
total_loss_3nn = 0

# 2.2 - Computing Loss
for key in prediction:
    dist = distance.euclidean(coord_test[key], prediction[key])
    dist_3nn.append(dist)
    total_loss_3nn += dist

print(dist_3nn)
print(total_loss_3nn/20)


# 3 - PLOTTING THE DATA

x_nn = []
y_nn = []

x_3nn = []
y_3nn = []

x_test = []
y_test = []

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


prednn = pd.DataFrame(predictionn).transpose().rename(columns={0: 'x',1:'y'})
pred3nn = pd.DataFrame(prediction).transpose().rename(columns={0: 'x',1:'y'})
real = pd.DataFrame(coord_test).transpose().rename(columns={0: 'x',1:'y'})
train = pd.DataFrame(coord_train).transpose().rename(columns={0: 'x',1:'y'})



plt.scatter( x='x', y='y', data=train, c='g')
plt.scatter( x='x', y='y', data=real, c='b')
plt.show()
plt.scatter( x='x', y='y', data=real, c='b')
plt.scatter( x='x', y='y', data=pred3nn, c='c')
plt.show()