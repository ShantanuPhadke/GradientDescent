import numpy as np
#import matplotlib.pyplot as pyplot

myXData = np.loadtxt('ex2x.dat')
myYData = np.loadtxt('ex2y.dat')

'''
print('X variable data: ')
for xElement in myXData:
	print(xElement)
print('')
print('Y variable data: ')
for yElement in myYData:
	print(yElement)
print ("")
print ("")
'''

learning_rate = 0.07
#pyplot.plot(myXData, myYData, 'ro')

#Initial values of omega for the hypothesis function
omega = np.array([0,0])

#Make an array full of ones of the length of myXData
ones_array = np.zeros(len(myXData))
for index in range(len(myXData)):
	ones_array[index] = 1

#Want our final x-data array in the form array([ [1 x11] [1 x21] ... [1 xn1]])
x_data = np.array([])
for index in range(len(myXData)):
	current_row = np.array([ones_array[index], myXData[index]])
	if index == 0:
		x_data = current_row
	else:
		x_data = np.vstack((x_data, current_row))

#print(x_data)


#h(x^i) = omega[0] * 1 + omega[1]*x_value
def hypothesis(x_value):
	return omega[0]*x_value[0] + omega[1]*x_value[1]

def delta(x_list, y_list):
	my_sum = np.zeros(2)
	for index in range(len(y_list)):
		current_error = hypothesis(x_list[index])-y_list[index]
		#print current_error
		addition_term = current_error*x_list[index]
		my_sum+= addition_term
	#print my_sum
	my_sum = my_sum/len(y_list)
	return my_sum

#Case 1: Single Iteration of Gradient Descent
omega = omega - learning_rate * delta(x_data, myYData)

print ( "Value of omega[0] after 1 iteration: " + str(omega[0]) )
print ( "Value of omega[1] after 1 iteration: " + str(omega[1]) )

#Case 2: Running Gradient Descent till convergence of the array omega
iteration_count = 0
while(iteration_count < 1500):
	omega = omega - learning_rate * delta(x_data, myYData)
	iteration_count = iteration_count+1

print ( "Value of omega[0] after convergence of gradient descent: " + str(omega[0]) )
print ( "Value of omega[1] after convergence of gradient descent: " + str(omega[1]) )

print ("")
print ("")

print ( "Predicted height for age 3.5: " + str(hypothesis(np.array([1, 3.5]))) )
print ( "Predicted height for age 7: " + str(hypothesis(np.array([1, 7]))) )

