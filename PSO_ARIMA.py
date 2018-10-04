import csv
import matplotlib.pyplot as pyplot
import numpy as np
import random
from sklearn.metrics import mean_squared_error

# --- COST FUNCTION ------------------------------------------------------------+
def CostFunction(coe):
	global train
	n=len(train)
	tdata=[]
	sum=0.0
	for i in range(n-2):
		val=train[i]*coe[0]+train[i+1]*coe[1]+coe[2]
		if(i>1):
			val=val+(tdata[i-2]-train[i])*coe[3]+(tdata[i-1]-train[i+1])*coe[4]
		tdata.append(val)
		sum=sum+(tdata[i]-train[i+2])*(tdata[i]-train[i+2])/(n-2)
	return sum


#---------PREDICTION------------------------
def predict_psoarima(coe,num,no):
	global test
	global dataset
	z=[]
	for i in range(num):
		val=dataset[no-2+i]*coe[0]+dataset[no-2+i]*coe[1]+coe[2]
		if(i>1):
			val=val+(z[i-2]-dataset[no-2+i])*coe[3]+(z[i-1]-dataset[no-1+i])*coe[4]
		z.append(val)
	return  z


# --- MAIN ---------------------------------------------------------------------+

class Particle:
	def __init__(self, x0):
		self.position_i = []  # particle position
		self.velocity_i = []  # particle velocity
		self.pos_best_i = []  # best position individual
		self.err_best_i = -1  # best error individual
		self.err_i = -1  # error individual

		for i in range(0, num_dimensions):
			self.velocity_i.append(random.uniform(-1, 1))
			self.position_i.append(x0[i])

	# evaluate current fitness
	def evaluate(self, costFunc):
		self.err_i = costFunc(self.position_i)

		# check to see if the current position is an individual best
		if self.err_i < self.err_best_i or self.err_best_i == -1:
			self.pos_best_i = self.position_i
			self.err_best_i = self.err_i

	# update new particle velocity
	def update_velocity(self, pos_best_g):
		w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
		c1 = 1  # cognative constant
		c2 = 2  # social constant

		for i in range(0, num_dimensions):
			r1 = random.random()
			r2 = random.random()

			vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
			vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
			self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

	# update the particle position based off new velocity updates
	def update_position(self, bounds):
		for i in range(0, num_dimensions):
			self.position_i[i] = self.position_i[i] + self.velocity_i[i]

			# adjust maximum position if necessary
			if self.position_i[i] > bounds[i][1]:
				self.position_i[i] = bounds[i][1]

			# adjust minimum position if neseccary
			if self.position_i[i] < bounds[i][0]:
				self.position_i[i] = bounds[i][0]


class PSO():
	def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
		global num_dimensions
		global  pos_best_g
		num_dimensions = len(x0)
		err_best_g = -1  # best error for group
		pos_best_g = []  # best position for group

		# establish the swarm
		swarm = []
		for i in range(0, num_particles):
			swarm.append(Particle(x0))

		# begin optimization loop
		i = 0
		while i < maxiter:
			# print i,err_best_g
			# cycle through particles in swarm and evaluate fitness
			for j in range(0, num_particles):
				swarm[j].evaluate(costFunc)

				# determine if current particle is the best (globally)
				if swarm[j].err_i < err_best_g or err_best_g == -1:
					pos_best_g = list(swarm[j].position_i)
					err_best_g = float(swarm[j].err_i)

			# cycle through swarm and update velocities and position
			for j in range(0, num_particles):
				swarm[j].update_velocity(pos_best_g)
				swarm[j].update_position(bounds)
			i += 1

		# print final results
		print ('FINAL:')
		print (pos_best_g)
		print (err_best_g)




# --- RUN ----------------------------------------------------------------------+

initial = [0.01, 0.01,0.01,0.01,0.01]  # initial starting location [x1,x2...]
bounds = [(-10, 10), (-10, 10) ,(-10, 10) ,(-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]


# --- END ----------------------------------------------------------------------+
# loading data from excel fille
with open('Singapore Exchange Ltd.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    series=[]
    for row in reader:
        series.append(float(row['Close']))# select the Close row from full rows.
series= np.asarray(series)
# given data
global dataset
global train
global test
dataset = series
# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size

train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
global  pos_best_g
PSO(CostFunction, initial, bounds, num_particles=15, maxiter=150)

testPredictPlot=predict_psoarima(pos_best_g,test_size,train_size-1)

s=0.0
s1=0.0
num=0
Ms=np.mean(test)
# model fitting and prediction with testing data.
for t in range(len(test)):

	s=s+abs(test[t]-testPredictPlot[t])
	s1=s1+abs(Ms-test[t])
	num+=1
# calculation of RMSE
error = mean_squared_error(test, testPredictPlot)
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
pyplot.plot(testPredictPlot, 'r', label='Prediction Data', linewidth=1)

pyplot.grid()
pyplot.title("PSO_Arima algorithm")
pyplot.ylabel("Stock price")
pyplot.xlabel("Time")
pyplot.legend()
pyplot.show()
"""""
pyplot.plot(test)
pyplot.show()
pyplot.plot(testPredictPlot)
pyplot.show()
"""




