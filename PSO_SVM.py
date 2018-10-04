import csv
import matplotlib.pyplot as pyplot
import numpy as np
from pyswarm import pso
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn import preprocessing as pre


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
    global X_train_scaled
    global trainY
    clf = svm.SVR(kernel='rbf', C=x[0], gamma= x[1])
    clf.fit(X_train_scaled, trainY)
    a=clf.predict(X_train_scaled)
    mse = np.sqrt(mean_squared_error(trainY, a))
    return  mse


# loading data from excel fille
with open('Singapore Exchange Ltd.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    series=[]
    for row in reader:
        series.append(float(row['Open']))
series= np.asarray(series)
global dataset
global X_train_scaled
global trainY
global X_test_scaled
global testY
# given data
dataset = series
# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
#create the training and testind data
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create SVM model
scaler = pre.StandardScaler().fit(trainX)
X_train_scaled = scaler.transform(trainX)
X_test_scaled = scaler.transform(testX)

lb = [10, 0.0001]
ub = [100,0.1]
xopt, fopt = pso(fitness, lb, ub,swarmsize=15)
clf = svm.SVR(kernel='rbf', C=xopt[0], gamma=xopt[1])
clf.fit(X_test_scaled, testY)
a = clf.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(testY, a))
s=0.0
s1=0.0
num=0
Ms=np.mean(testY)
# model fitting and prediction with testing data.
for t in range(len(testY)):

	s=s+abs(testY[t]-a[t])
	s1=s1+abs(Ms-testY[t])
	num+=1

MAE=(s/num)
MAPE=(s/num)*100
Rsquare=1-s/s1
print('MAE: %.4f ' % MAE)
print('MAPE: %.4f ' % MAPE)
print('RMSE: %.4f ' % rmse)
print('Rsquare: %.4f ' % Rsquare)


pyplot.plot(testY, 'b', label='Real Data', linewidth=1)
    #plt.show()#
pyplot.plot(a, 'r', label='Prediction Data', linewidth=1)

pyplot.grid()
pyplot.title("PSO_SVM algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""
pyplot.plot(testY)
pyplot.show()
pyplot.plot(a)
pyplot.show()
"""
