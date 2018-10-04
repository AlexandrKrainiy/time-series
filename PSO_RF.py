import csv
import matplotlib.pyplot as pyplot
import numpy as np
from pyswarm import pso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pre
import pandas as pd

# getting training and testing data from full dataset
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

def fitness(x):
    global dataset
    global X
    global y
    clf = RandomForestClassifier(n_estimators=(int)(x[0]),n_jobs=(int)(x[1]))
    clf.fit(X, y)
    a=clf.predict(X)
    mse = np.sqrt(mean_squared_error(y, a))
    return  mse


# loading data from excel fille
with open('Singapore Exchange Ltd.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    series=[]
    for row in reader:
        series.append(float(row['Open']))
series= np.asarray(series)
global dataset
global X
global y
global testX
global testY
# given data
dataset = series
X =  np.zeros(720).reshape(-1,1)
y = np.zeros(720).reshape(-1,1)
Y = np.zeros(150).reshape(-1,1)
for i in range(np.size(y, axis=0)):
    X[i] = dataset[i]
    y[i] = i
for i in range(np.size(Y, axis=0)):
    Y[i] = dataset[i+720]


lb = [50, 2]
ub = [100,5]
xopt, fopt = pso(fitness, lb, ub,swarmsize=15,maxiter=10)
print(xopt)
clf = RandomForestClassifier(n_estimators=(int)(xopt[0]), n_jobs=(int)(xopt[1]))
clf.fit(X, y)
a = clf.predict(Y)
s=0.0
s1=0.0
s2=0.0
num=0
Ms=np.mean(Y)
Z=[]
# model fitting and prediction with testing data.
for t in range(len(Y)):
    index=(int)(a[t])
    s2=s2+(Y[t]-X[index])*(Y[t]-X[index])
    s=s+abs(Y[t]-X[index])
    s1=s1+abs(Ms-Y[t])
    Z.append(X[index])
    num+=1
rmse = np.sqrt(s2/num)
MAE=(s/num)
MAPE=(s/num)*100
Rsquare=1-s/s1
print('MAE: %.6f ' % MAE)
print('MAPE: %.6f ' % MAPE)
print('RMSE: %.6f ' % rmse)
print('Rsquare: %.6f ' % Rsquare)

pyplot.plot(Y, 'b', label='Real Data', linewidth=1)
    #plt.show()#
pyplot.plot(Z, 'r', label='Prediction Data', linewidth=1)

pyplot.grid()
pyplot.title("RF algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""
pyplot.plot(Y)
pyplot.show()
pyplot.plot(Z)
pyplot.show()
"""