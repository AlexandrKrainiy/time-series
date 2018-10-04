from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import csv
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
        series.append(float(row['Close']))# select the Close row from full rows.
series= np.asarray(series)

dataset = series

# real tine series values
true_trend=np.zeros(720)
for i in range(720):
    true_trend[i]=dataset[i]


# prediction from KNN model with testing data.
def recover_trend():
    y = np.zeros(720)
    x = np.zeros(720).reshape(-1,1)
    for i in range (np.size(y,axis=0)):
        y[i]=dataset[i]
        x[i]=i

    #KNN model
    model = KNeighborsRegressor()
    model.fit(x, y)# fitting part
    # prediction
    B=np.arange(720).reshape(-1, 1)
    rec_trend = model.predict(B)
    return x, y, rec_trend


def main():
    # prediction
    x, y, rec_trend = recover_trend()
    s = 0.0
    s1 = 0.0
    num = 0
    Ms=np.mean(y)
    for i in range(len(x)):
        s=s+abs(rec_trend[i]-true_trend[i])
        s1=s1+abs(true_trend[i]-Ms)
        num+=1
    # plotting result
    plt.plot(true_trend, 'b', label='Real Data', linewidth=1)
    #plt.show()
    plt.plot(rec_trend, 'r', label='Prediction Data', linewidth=2)

    plt.grid()
    plt.title("KNN algorithm")
    plt.ylabel("Stock price")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
    rmse = np.dot((rec_trend - true_trend).T, (rec_trend - true_trend))
    n = 720
    rmse = rmse / n
    MAE = (s / num)
    MAPE = (s / num) * 100
    Rsquare = 1 - s / s1
    print('MAE: %.4f ' % MAE)
    print('MAPE: %.4f ' % MAPE)
    print('RMSE: %.4f ' % rmse)
    print('Rsquare: %.4f ' % Rsquare)


if __name__ == '__main__':
    main()