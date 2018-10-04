import csv
import matplotlib.pyplot as pyplot
from sklearn import svm
import numpy as np
from sklearn import preprocessing as pre
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
# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test))
#create the training and testind data
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# create SVM model
scaler = pre.StandardScaler().fit(trainX)
X_train_scaled = scaler.transform(trainX)
X_test_scaled = scaler.transform(testX)
SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.001).fit(X_train_scaled,trainY)
predict_y_array = SVR_model.predict(X_test_scaled)
Ms=np.mean(testY)
# prediction of SVM model with testing data
m=len(X_train_scaled)
Score=SVR_model.score(X_test_scaled,testY)
testPredictPlot = dataset
s=0.0
s1=0.0
num=0
for i in range (m+ (1 * 2) + 1, len(dataset) - 1):
	testPredictPlot[i]=predict_y_array[i-(m+ (1 * 2) + 1)]
	s=s+abs(testPredictPlot[i]-testY[i-(m+ (1 * 2) + 1)])
	s1=s1+abs(Ms-testY[i-(m+ (1 * 2) + 1)])
	num=num+1
# calculation of RMSE
rmse=np.dot((predict_y_array-testY).T,(predict_y_array-testY))
n=len(predict_y_array)
rmse=rmse/n
MAE=(s/num)
MAPE=(s/num)*100
Rsquare=1-s/s1
print('MAE: %.4f ' % MAE)
print('MAPE: %.4f ' % MAPE)
print('RMSE: %.4f ' % rmse)
print('Rsquare: %.4f ' % Rsquare)
# plot baseline and predictions
pyplot.plot(testY, 'b', label='Real Data', linewidth=1)
    #plt.show()#
pyplot.plot(predict_y_array, 'r', label='Prediction Data', linewidth=2)

pyplot.grid()
pyplot.title("SVM algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""
pyplot.plot(X_test_scaled)
pyplot.show()
pyplot.plot(predict_y_array)
pyplot.show()
"""