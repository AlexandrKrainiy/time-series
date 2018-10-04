import csv
import matplotlib.pyplot as pyplot
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# getting training and testing data from full dataset
with open('UOB1(3).csv') as csvfile:
    reader = csv.DictReader(csvfile)
    series=[]
    for row in reader:
        series.append(float(row['Close']))
series= np.asarray(series)
X = series
# seperation of training and testing data
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
s=0.0
s1=0.0
num=0
Ms=np.mean(test)
# model fitting and prediction with testing data.
for t in range(len(test)):
	model = ARIMA(history, order=(3,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	s=s+abs(test[t]-predictions[t])
	s1=s1+abs(Ms-test[t])
	num+=1
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
# calculation of RMSE
error = mean_squared_error(test, predictions)
RMSE=np.sqrt(error)
MAE=(s/num)
MAPE=(s/num)*100
Rsquare=1-s/s1
print('MAE: %.4f ' % MAE)
print('MAPE: %.4f ' % MAPE)
print('RMSE: %.4f ' % RMSE)
print('Rsquare: %.4f ' % Rsquare)
pyplot.plot(test, 'b', label='Real Data', linewidth=1)
    #plt.show()#
pyplot.plot(predictions, 'r', label='Prediction Data', linewidth=1)

pyplot.grid()
pyplot.title("Arima algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""

# plot
pyplot.plot(test)
pyplot.show()
pyplot.plot(predictions, color='blue')
pyplot.show()
"""