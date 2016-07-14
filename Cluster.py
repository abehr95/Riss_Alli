import sys, math, random
from decimal import*
#from matplotlib.dates import date2num
from datetime import date, datetime, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Cluster:

	def __init__(self, data):
		"""data = {group:[rvalue1, rvalue2,...], ...}"""
		self.data = data
		self.groups = []
		self.rvalues = []
		self.centroid = {}
		self.rclusters = {}
		self.gclusters = {}

	def data_read(self):
		for group in self.data:
			self.groups.append(group)
			if type(self.data[group]) == list:
				if all(item >= -.01 for item in self.data[group]):
					self.rvalues.append(self.data[group])
				else:
					print "thrown out: ", self.data[group]
			else:
				if self.data[group] >= -.01:
					self.rvalues.append(self.data[group])
				else:
					print "thrown out: ", self.data[group]

	def compute_distance(self,data_type, point):
		temp = {}
		L = sorted(data_type.keys(),reverse = False)
		for i in range(len(L)):
			val = L[i]
			centroid_point = data_type[val]
			distance = 0.0
			for j in range(len(centroid_point)):
				distance += pow((centroid_point[j]-point[j]),2)
			temp[val]= distance
		nearest_centroid = min(temp, key = temp.get)
		return nearest_centroid


	def compute_centroid(self,cluster,centroid):
		m = sorted(cluster.keys(), reverse = False)
		temp = []
		temp_holder = {}
		for i in range(len(m)):
			cluster_key = m[i]
			current_cluster = cluster[cluster_key]
			temp_holder = []
			new_centroid = []
			for j in range(len(current_cluster)):
				current_point = current_cluster[j]
				for k in range(len(current_point)):
					val_to_add = current_point[k]
					if j == 0:
						temp_holder.append(val_to_add)
					else:
						curr_sum = temp_holder[k]
						new_sum = val_to_add + curr_sum
						temp_holder[k]= new_sum
			for point in range(len(temp_holder)):
				mean_val = float(temp_holder[point]/len(current_cluster))
				new_centroid.append(mean_val)
			old_centroid = centroid[cluster_key]
			centroid_diff = []
			for x in range(len(new_centroid)):
				curr_diff = abs(new_centroid[x]-old_centroid[x])
				centroid_diff.append(curr_diff)
			max_diff = max(centroid_diff)
			temp.append(max_diff)
			centroid[cluster_key]= new_centroid
		biggest_diff = max(temp)
		return biggest_diff

	def kmeans(self,points,groups,k,cutoff):
		initial_guess = random.sample(points,k)
		#print "initial_guess: " + str(initial_guess)
		clusters = {}
		group_clusters = {}
		for i in range(k):
			self.centroid[i+1]= initial_guess[i]

		num_iters = 1
		biggest_shift = cutoff
		while biggest_shift >= cutoff and num_iters < 1000:
			for i in range(k):
				clusters[i+1]= []
				group_clusters[i+1] = []
			for p in range(len(points)):
				curr_point = points[p]
				curr_group = groups[p]
				#print "curr_point: " + str(curr_point)
				near_cluster = self.compute_distance(self.centroid,curr_point)
				clusters[near_cluster].append(curr_point)
				group_clusters[near_cluster].append(curr_group)

			biggest_shift = self.compute_centroid(clusters,self.centroid)
			num_iters += 1
			#print "num_iters: " + str(num_iters)
		return clusters, group_clusters

	def data_write(self,data_type,sheet_name):
		"""Takes dictionary (key = percentile, val = travel_rate) and user specified
		excel worksheet name as input arguments and outputs those dicitonary values"""

		c = 0
		r = 0
		write_along_column(sheet_name, ['r2_39_9', 'r2_9_10'],0,c+1)
		for r in xrange(len(data_type)):
			temp = data_type[r]
			#print "val" + str(val)
			#temp = data_type[perc]
			#print "d, e, f: " + str(d) + str(e) + str(f)
			sheet_name.write(r+1, 1, temp[0])
			sheet_name.write(r+1, 2, temp[1])
			#book_results.write(r+1, 3, z)
			#book_results.write(r+1, 4, freq)
			r += 1

	def main(self):
		self.data_read()
		clusters, group_clusters = self.kmeans(self.rvalues,self.groups,6,0.000000000001)
		self.rclusters, self.gclusters = clusters , group_clusters
		#print "clusters: " + str(clusters)
		#print "group_clusters: " + str(group_clusters)
		self.plot_clusters()

		return group_clusters

	def plot_clusters(self):
		colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
		cluster_names = ["Cluster " + str(num) for num in self.rclusters]
		if len(self.rvalues[0]) == 1:
			plots = []
			i = 0
			for cluster in self.rclusters:
				x = np.linspace(0,len(self.rclusters[cluster]),len(self.rclusters[cluster])+1)
				y = np.array([x[0] for x in self.rclusters[cluster]])
				plots.append(plt.scatter(x,y,c=colors[i]))
				i += 1
				if i >= len(colors):
					i = 0
			plt.legend(plots,cluster_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
			plt.ylabel('R1')
			plt.title('Clusters')

		if len(self.rvalues[0]) == 2:
			plots = []
			i = 0
			for cluster in self.rclusters:
				#print self.rclusters[cluster]
				x = np.array([el[0] for el in self.rclusters[cluster]])
				y = np.array([el[1] for el in self.rclusters[cluster]])
				#print x, y
				#print len(x), len(y)
				plots.append(plt.scatter(x,y,c=colors[i]))
				i += 1
				if i >= len(colors):
					i = 0
			plt.legend(plots,cluster_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
			plt.xlabel('R1')
			plt.ylabel('R2')
			plt.title('Clusters')

		if len(self.rvalues[0]) == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			plots = []
			i = 0
			for cluster in self.rclusters:
				x = np.array([el[0] for el in self.rclusters[cluster]])
				y = np.array([el[1] for el in self.rclusters[cluster]])
				z = np.array([el[2] for el in self.rclusters[cluster]])
				plots.append(ax.scatter(x, y, z,c=colors[i]))
				i += 1
				if i >= len(colors):
					i = 0
				ax.legend(plots,cluster_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
				ax.set_xlabel('R1')
				ax.set_ylabel('R2')
				ax.set_zlabel('R3')
				ax.set_title('Clusters')
		plt.show()

	def save_plot(self, filename):
		colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
		cluster_names = ["Cluster " + str(num) for num in self.rclusters]
		if len(self.rvalues[0]) == 1:
			plots = []
			i = 0
			for cluster in self.rclusters:
				x = np.linspace(0,len(self.rclusters[cluster]),len(self.rclusters[cluster])+1)
				y = np.array([x[0] for x in self.rclusters[cluster]])
				plots.append(plt.scatter(x,y,c=colors[i]))
				i += 1
				if i >= len(colors):
					i = 0
			plt.legend(plots,cluster_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
			plt.ylabel('R1')
			plt.title('Clusters')

		if len(self.rvalues[0]) == 2:
			plots = []
			i = 0
			for cluster in self.rclusters:
				#print self.rclusters[cluster]
				x = np.array([el[0] for el in self.rclusters[cluster]])
				y = np.array([el[1] for el in self.rclusters[cluster]])
				#print x, y
				#print len(x), len(y)
				plots.append(plt.scatter(x,y,c=colors[i]))
				i += 1
				if i >= len(colors):
					i = 0
			plt.legend(plots,cluster_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
			plt.xlabel('R1')
			plt.ylabel('R2')
			plt.title('Clusters')

		if len(self.rvalues[0]) == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			plots = []
			i = 0
			for cluster in self.rclusters:
				print 
				x = np.array([el[0] for el in self.rclusters[cluster]])
				y = np.array([el[1] for el in self.rclusters[cluster]])
				z = np.array([el[2] for el in self.rclusters[cluster]])
				plots.append(ax.scatter(x, y, z,c=colors[i]))
				i += 1
				if i >= len(colors):
					i = 0
				ax.legend(plots,cluster_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
				ax.set_xlabel('R1')
				ax.set_ylabel('R2')
				ax.set_zlabel('R3')
				ax.set_title('Clusters')

		plt.savefig(filename)


