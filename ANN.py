import csv
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.layers import Dense
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
with open('UOB1.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    series=[]
    for row in reader:
        series.append(float(row['Close']))# select the Close row from full rows.
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
#create ANN model.
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
# fitting the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=2, verbose=2)


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# prediction with testind data
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
Ms=np.mean(testPredict)
m=len(trainPredict)
# shift train predictions for plotting
trainPredictPlot = dataset
s=0.0
s1=0.0
for i in range (look_back, m+look_back):
	trainPredictPlot[i]=trainPredict[i-look_back][0]

num=0
testPredictPlot = dataset

for i in range (m+ (look_back * 2) + 1, len(dataset) - 1):
	testPredictPlot[i]=testPredict[i-(m+ (look_back * 2) + 1)][0]
	s=s+abs(testPredict[i-(m+ (look_back * 2) + 1)][0]-testY[i-(m+ (look_back * 2) + 1)])
	s1 = s1 + abs(Ms - testY[i - (m + (look_back * 2) + 1)])
	num+=1
MAE=(s/num)
MAPE=(s/num)*100
Rsquare=1-s/s1
print('MAE: %.2f ' % MAE)
print('MAPE: %.2f ' % MAPE)
print('RMSE: %.2f ' % np.sqrt(testScore))
print('Rsquare: %.2f ' % Rsquare)
# plot baseline and predictions
#pyplot.plot(dataset)
#pyplot.show()
pyplot.plot(testY, 'b', label='Real Data', linewidth=1)
    #plt.show()#
pyplot.plot(testPredict, 'r', label='Prediction Data', linewidth=2)

pyplot.grid()
pyplot.title("ANN algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""
pyplot.plot(testX)
pyplot.show()
pyplot.plot(testPredict)
pyplot.show()
"""