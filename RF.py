from sklearn.ensemble import RandomForestRegressor
import csv
import matplotlib.pyplot as pyplot
import numpy as np


# getting training and testing data from full dataset
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)
# loading data from excel fille
with open('Singapore Exchange Ltd.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    series=[]
    for row in reader:
        series.append(float(row['Close']))
series= np.asarray(series)
# given data
dataset = series
X =  np.zeros(720).reshape(-1,1)
y = np.zeros(720).reshape(-1,1)
for i in range(np.size(y, axis=0)):
    X[i] = dataset[i]
    y[i] = i
Ms=np.mean(X)
s=0.0
s1=0.0
num=0
# randomforecast regression.
regressor = RandomForestRegressor(n_estimators=150, min_samples_split=2)
# model fitting
regressor.fit(y, X)
# prediction of model
Ppred=regressor.predict(y)
for i in range(len(Ppred)):
    s=s+abs(Ppred[i]-dataset[i])
    s1=s1+abs(Ms-dataset[i])
    num+=1
# calculation of RMSE
rmse=np.dot((Ppred-dataset[:720]).T,(Ppred-dataset[:720]))
n=len(Ppred)
rmse=rmse/n
MAE=(s/num)
MAPE=(s/num)*100
Rsquare=1-s/s1
print('MAE: %.4f ' % MAE)
print('MAPE: %.4f ' % MAPE)
print('RMSE: %.4f ' % rmse)
print('Rsquare: %.4f ' % Rsquare)
pyplot.plot(X, 'b', label='Real Data', linewidth=1)
    #plt.show()#
pyplot.plot(Ppred, 'r', label='Prediction Data', linewidth=2)

pyplot.grid()
pyplot.title("RF algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""
pyplot.plot(X)
pyplot.show()
pyplot.plot(Ppred)
pyplot.show()
"""