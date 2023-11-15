# Utility class 

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d
import math
import scipy
import random
import vispy.scene
from vispy.scene import visuals

class integrate():
	"""This class applies vairous integration methods and give a comparitive graph"""
	def __init__(self, x_input, step):
		self.x_input = x_input
		self.y_init = -1
		self.x_init = 1
		self.step = -1*step
		self.results = {}
	
	def diff(self,x,y):
		# calculates derivative at x and y
		d = 1/((x**2)*(1-y))
		return d

	def analytical(self):
		# Calculates analytical solution of the differential equation
		y_analytical = np.zeros(self.x_input.shape) 
		for pos,i in enumerate(self.x_input):
			if pos == 0:
				y_analytical[pos] = -1
			elif pos == 20:
				y_analytical[pos] = -1*np.inf
			else:
				x = i
				y_analytical[pos] = 1 - np.sqrt((2/x) + 2)
		return y_analytical

	def euler(self):
		""" This Solves the differential equation by using:
			y_n = y_(n-1) + h*y'
		"""
		y_euler = np.zeros(self.x_input.shape)
		for pos,i in enumerate(self.x_input):
			if pos == 0:
				y_euler[pos] = -1
			else:
				der = self.diff(self.x_input[pos-1],y_euler[pos-1])
				y_euler[pos] = y_euler[pos-1] + self.step*der
		return y_euler

	def runge_kutta(self):
		""" This Solves the differential equation by using:

			y_n = y_(n-1) + k1/6 + k2/3 + k3/3 + k4/6

		"""
		y_rk = np.zeros(self.x_input.shape)
		for pos,i in enumerate(self.x_input):
			if pos == 0:
				y_rk[pos] = -1
			else:
				x,y = self.x_input[pos-1],y_rk[pos-1]
				k1 = self.step*self.diff(x,y)
				k2 = self.step*self.diff(x+(self.step/2),y+(k1/2))
				k3 = self.step*self.diff(x+(self.step/2),y+(k2/2))
				k4 = self.step*self.diff(x+self.step,y+k3)
				y_rk[pos] = y_rk[pos-1] + (k1/6) + (k2/3) + (k3/3) + (k4/6)   
		return y_rk


	def richardson_extrapolation(self):
		""" This Solves the differential equation by using:

			y_n = 

		"""
		print('evaluating richardson')
		n = 100
		h = self.step/n
		y_riche = np.zeros(self.x_input.shape)
		for pos,i in enumerate(self.x_input):
			if pos == 0:
				y_riche[pos] = -1
			else:
				x,z_n_1 = self.x_input[pos-1],y_riche[pos-1]
				z_n = z_n_1 + h*self.diff(x,z_n_1)
				for j in range(n-1):
					a = z_n_1 + 2*h*self.diff(x+(j+1)*h,z_n)
					z_n_1 = z_n
					z_n = a
				y_riche[pos] = 0.5*(z_n + z_n_1 + h*self.diff(x+self.step,z_n))    
		
		return y_riche
	
	def error(self,method1,method2):
		return [np.mean(np.abs(method1[:-1]-method2[:-1])),np.std(method1[:-1]-method2[:-1])]
	
	def integrate(self,methods):
		if 1 in methods:
			analytical_ans = self.analytical()
			self.results['analytical'] = [analytical_ans,1,self.error(analytical_ans,analytical_ans)]

		if 2 in methods:
			euler_ans = self.euler()
			self.results['euler'] = [euler_ans,2,self.error(euler_ans,self.analytical())]

		if 3 in methods:
			rk_ans = self.runge_kutta()
			self.results['runge-kutta'] = [rk_ans,3,self.error(rk_ans,self.analytical())]

		if 4 in methods:
			re_ans = self.richardson_extrapolation()
			self.results['richardson-extrapolation'] = [re_ans,4,self.error(re_ans,self.analytical())]

		print(self.results)


	def plot(self):
		for method in self.results:
			plt.plot(self.x_input[:-1], self.results[method][0][:-1], 
				     label = method, marker='o', 
				     alpha = 1/self.results[method][1])
			plt.legend() 
		plt.show() 


class extract():
	"""Extracts plane from given point cloud data"""
	def __init__(self):
		self.method = 0
		self.error_max = 0
		self.num_points = 3
		self.best_plane = 0
		self.prob_max = 0.99
		self.max_inlier = 0
		self.error_max = 0.01
		self.planes = []

	def load_file(self,file_location):
		self.data = np.loadtxt(file_location)
		self.og_data = self.data
		print(self.data.shape)
		self.data = self.data[~np.all(self.data == 0, axis=1)]
		print(np.var(self.data))

	
	def plot_data_2(self):
		

		
		# generate data
		canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
		view = canvas.central_widget.add_view()
		pos = self.og_data
		# create scatter object and fill in the data
		scatter = visuals.Markers()
		scatter.set_data(pos, edge_width=0, face_color=(1, 0, 0, .5), size=5)
		view.add(scatter)
		view.camera = 'arcball'  # or try 'arcball'
		
		for best_plane in self.planes:
			model = best_plane['best plane']
			[x_min,y_min,z_min] = best_plane['range'][0]
			[x_max,y_max,z_max] = best_plane['range'][1]
			x = np.linspace(x_min,x_max,100)
			y = np.linspace(y_min,y_max,100)
			x, y = np.meshgrid(x, y)
			z = (1 - model[0] * x - model[1] * y) / model[2]
			p1 = visuals.SurfacePlot(x = x,y = y, z=z, color=(0.5+random.uniform(-0.5,0.5),
															  0.5+random.uniform(-0.5,0.5),
															  0.5+random.uniform(-0.5,0.5), 
															  0.7))
			view.add(p1)
		axis = visuals.XYZAxis(parent=view.scene)
		vispy.app.run()

	def plot_data(self):
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2])
		x = np.linspace(np.min(self.data),np.max(self.data),100)
		y = np.linspace(np.min(self.data),np.max(self.data),100)
		x, y = np.meshgrid(x, y)
		#print(self.best_plane[0].shape)
		z = (1 - self.best_plane[0][0] * x - self.best_plane[1][0] * y) / self.best_plane[2][0]
		ax.plot_surface(x, y, z, alpha=0.5,cmap=cm.coolwarm,color='red')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		
		plt.show()

	def svd_solver(self,p):
		A_T_A = np.dot(p.T, p)
		eigen_values, eigen_vectors = np.linalg.eig(A_T_A)
		eigen_vector_min = eigen_vectors[:, np.argmin(eigen_values)]
		return eigen_vector_min


	def ls_solver(self,A,B):
		x,_,_,_ = np.linalg.lstsq(A,B)
		return np.array(x)

	def plane_eq(self,sample_points):

		matrix_B = np.ones((sample_points.shape[0],1))
		plane_coeff = self.ls_solver(sample_points,matrix_B)
		return plane_coeff

	def cal_mean_error(self, matrix_A):
		
		output = np.ones((matrix_A.shape[0],1))
		normalising_coeff = np.sqrt((self.plane_coeff[0])**2+(self.plane_coeff[1])**2+(self.plane_coeff[2])**2)
		distance = np.abs(np.dot(matrix_A,self.plane_coeff) - output)/(normalising_coeff[0])
		
		return np.mean(distance)

	def count_inlier_2(self, sm_points):
		set_A = set(map(tuple, self.data))
		set_B = set(map(tuple, sm_points))

		# Find the difference between set_A and set_B
		result_set = set_A - set_B

		# Convert the result set back to a NumPy array
		matrix_A = np.array(list(result_set))
		output = np.ones((matrix_A.shape[0],1))
		normalising_coeff = np.sqrt((self.plane_coeff[0])**2+(self.plane_coeff[1])**2+(self.plane_coeff[2])**2)
		distance = np.abs(np.dot(matrix_A,self.plane_coeff) - output)/(normalising_coeff[0])
		inliers, dist_inliars = matrix_A[distance[:,0] < self.error_max], distance[distance[:,0] < self.error_max]
		inliers = np.vstack((sm_points,inliers))
		num_inlier = inliers.shape[0]

		set_B = set(map(tuple, inliers))

		# Find the difference between set_A and set_B
		result_set = set_A - set_B

		# Convert the result set back to a NumPy array
		rest_points = np.array(list(result_set))
		
		return inliers, num_inlier, np.mean(dist_inliars), rest_points

	def count_inlier(self):

		matrix_A = self.data[self.num_points:,:]
		output = np.ones((matrix_A.shape[0],1))
		normalising_coeff = np.sqrt((self.plane_coeff[0])**2+(self.plane_coeff[1])**2+(self.plane_coeff[2])**2)
		distance = np.abs(np.dot(matrix_A,self.plane_coeff) - output)/(normalising_coeff[0])
		inliers, dist_inliars = self.data[self.num_points:,:][distance[:,0] < self.error_max], distance[distance[:,0] < self.error_max]
		num_inlier = inliers.shape[0]
		rest_points = self.data[self.num_points:,:][distance[:,0] > self.error_max]

		
		return inliers, num_inlier, np.mean(dist_inliars), rest_points
	
	def ransac_1(self):

		max_iterations = 150
		i = 0
		err_now = np.inf
		max_inliers = 0
		best_plane = 0
		max_inlier = 0
		max_mean_error = 0
		rest_points = 0
		inlier_points = 0
		min_max_range = 0
		while(i < max_iterations):
			#print('i = ', i)
			np.random.shuffle(self.data)
			sample_points = self.data[0:self.num_points,:]

			# Fit Plane
			self.plane_coeff = self.plane_eq(sample_points)

			# Quantify Plane
			inliers, num_inlier, mean_error, rest_p = self.count_inlier()
			if num_inlier > max_inliers:
				max_inliers = num_inlier
				inlier_points = inliers
				best_plane = self.plane_coeff
				max_inlier = max_inliers
				min_max_range = [np.min(inliers,0), np.max(inliers,0)]
				rest_points = rest_p
				max_mean_error = mean_error

				# From here we have to comment
				print('og',i,max_inlier,max_mean_error, min_max_range)
				self.plane_coeff = self.plane_eq(np.vstack((sample_points,inliers)))
				inliers,num_inlier,mean_error, rest_p = self.count_inlier_2(np.vstack((sample_points,inliers)))
				if self.cal_mean_error(inliers) < max_mean_error:
					max_inliers = num_inlier
					inlier_points = inliers
					best_plane = self.plane_coeff
					max_inlier = max_inliers
					min_max_range = [np.min(inliers,0), np.max(inliers,0)]
					rest_points = rest_p
					max_mean_error = self.cal_mean_error(inlier_points)
				print('After',i,max_inlier,max_mean_error, min_max_range)
				# Till Here

			if i % 20 == 0:
				print('i = ', i,'num_inlier', max_inlier, 'mean_error', max_mean_error)

			i = i + 1
		print([best_plane,max_inlier,max_mean_error, min_max_range])
		self.data = rest_points

		return [best_plane,max_inlier,max_mean_error, min_max_range]
		

	def ransac(self, n = 10):
		k = 0
		num_try = 1
		while(k < n and num_try < 30 and self.data.shape[0]>0.01*self.og_data.shape[0]):
			print(k)
			[best_plane,max_inlier,max_mean_error, min_max_range] = self.ransac_1()
			if (max_mean_error < 0.1 or max_inlier > 25000) :
				self.planes.append({'best plane':best_plane,
				                'max_inlier':max_inlier,
				                'max_mean error':max_mean_error,
				                'range': min_max_range,
				                'inlier rartio': max_inlier/self.og_data.shape[0]})
				k = k+1
			num_try = num_try + 1
			print(num_try)
		print(self.planes)
