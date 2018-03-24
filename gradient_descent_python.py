from numpy import *

def compute_error_for_line_given_points(b, m, points):
	totalError=0
	for i in range(len(points)):
		x=points[i,0]
		y=points[i,1]
		err=(y-(m*x + b))**2
		totalError += err

	return totalError/float(len(points))

def step_gradient(b,m,points,learning_rate):
	N=len(points)
	b_grad=0
	m_grad=0
	for i in range(N):
		x=points[i,0]
		y=points[i,1]

		err = y -(m*x + b)


		b_grad += err
		m_grad += err*x


	N=float(N)


	new_b=b- learning_rate*(2/N)*(-1)*b_grad
	new_m=m- learning_rate*(2/N)*(-1)*m_grad
	return [new_b,new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b,m=step_gradient(b,m,array(points),learning_rate)

	return [b,m]

def compute_accuracy(b,m,points):
	accuracy=0
	N=len(points)
	for i in range(N):
		x=points[i,0]
		y=points[i,1]
		y_pred = m*x +b
		diff= abs(y_pred - y)
		rel_diff= diff/y
		accuracy += (1 - rel_diff)

	N=float(N)

	return (accuracy/N) * 100



def run():
	points=genfromtxt('data.csv',delimiter=',')
	learning_rate = 0.0001
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess

	num_iterations = 1000

	[b,m]=gradient_descent_runner(points,initial_b,initial_m,learning_rate, num_iterations)
	print("Accuracy {} %".format(compute_accuracy(b,m,array(points))))
	print("weight {}".format(m))
	print("bias {}".format(b))

if __name__=='__main__':
	run()
