import numpy as np
import json
import sys
import csv
from os import walk
import ast
from datetime import datetime
from datetime import timedelta
from fastkml import kml
from shapely.geometry import Point, LineString, Polygon
from Graph import CreateGraph
csv.field_size_limit(sys.maxsize)

trip_differentiator = 15
#unix time: import datetime library

class FileManage:

	def __init__(self, file_type, path):
		self.file_type = file_type
		self.file_data = {}
		self.path = path
		self.all_files = []

	def get_files(self):
		f = []
		for (dirpath, dirnames, filenames) in walk(self.path):
			f.extend(filenames)
			break
		g = []
		for el in f:
			self.all_files.append(el)
			if self.file_type in el:
				if 'SortedTrips' not in el:
					g.append(el)
		return g

	def add_data(self,sensor, data):
		if sensor not in self.file_data.keys():
			self.file_data[sensor] = data
		else:
			for id_ in data.keys():
				if id_ not in self.file_data[sensor].keys():
					self.file_data[sensor][id_] = data[id_]
				else:
					for time in data[id_]:
						if time not in self.file_data[sensor][id_]:
							self.file_data[sensor][id_] = self.file_data[sensor][id_].append(data[id_])

	def load_dat(self,x):
		array= np.loadtxt(x, dtype = 'str', delimiter='\n')
		data = []
		for el in array:
			d = json.loads(el)
			data.append(d)
		return data

	def is_empty(self,any_structure):
	    if any_structure:
	        return False
	    else:
	        return True

	def get_id_earliest_time(self,_data):
		weakest_signal = {}
		for entry in _data:
			if self.is_empty(entry["data"]) == False:
				_id = entry["addr"]
				time = entry["time"] + entry["data"][0][0]
				if weakest_signal.has_key(_id):
					times = weakest_signal[_id] 
				else:
					times = []
				times.append(time)
				weakest_signal[_id] = times
		return weakest_signal

	def load_all_files(self):
		files = self.get_files()
		if self.file_type == ".dat":
			for file_ in files:
				data = self.load_dat(file_)
				times = self.get_id_earliest_time(data)
				print "Earlist Time Retrieved for ", file_
				if self.is_empty(times) == False:
					self.file_data[float(file_[:-4])] = times
		elif self.file_type == ".csv":
			for file_ in files:
				if len(file_) == 7:
					print "Loading", file_
					self.file_data[float(file_[:-4])] = self.load_csv(file_)

		else:
			raise ValueError('Cannot load file type: ' + self.file_type) 


	def load_csv(self,x):
		id_data = {}
		with open(x, 'rb') as f:
			reader_ = csv.reader(f, delimiter=';')
			id_times = list(reader_)
			for data in id_times:
				time_list = ast.literal_eval(data[1])
				#time_id = [[time,x[:-4]] for time in time_list]
				id_data[data[0]]= time_list
		return id_data

	def get_data(self):
		return self.file_data

	def save_all(self):
		for sensor in self.file_data.keys():
			id_data = self.file_data[sensor]
			with open(str(int(sensor)) + ".csv", 'wb') as f:
			    writer = csv.writer(f)
			    writer = csv.writer(f, delimiter=';')
			    for row in id_data.iteritems():
			        writer.writerow(row)

	def save_dict(self,dict_, name, del_):
		with open(name + ".csv", 'wb') as f:
		    writer = csv.writer(f)
		    writer = csv.writer(f, delimiter=del_)
		    for row in dict_.iteritems():
		        writer.writerow(row)

	def get_all_files(self):
		return self.all_files


class RawDataManager:

	def __init__(self, path_):
		#path is file path where data is stored

		#trips will contain id:{sensor:[times]}, 

		#rawdata will be filled according to {sensor:{ida:[time1, time2,...], idb:[time1, time2, time3....], ...}, sensor 2:{}}

		self.detector_time = {}
		self.trips = {}
		self.path = path_
		self.raw_data = {}
		self.dat_object = "None"
		self.valid_connections = {}
		self.graph = "None"


	##uses FileManager class to retrieve sensor data 
	#from dat file to fill rawdata. save all files into csv form.
	def data_dat(self):
		self.dat_object = FileManage(".dat", self.path)
		self.dat_object.load_all_files()
		self.raw_data = self.dat_object.get_data()
		self.dat_object.save_all()

	#gets data to sill raw_data object from csv file. 
	def data_csv(self):
		self.dat_object = FileManage(".csv", self.path)
		self.dat_object.load_all_files()
		self.raw_data = self.dat_object.get_data()

	#trips data is created by sorting data into {id1:{sensor1:[time1, time2...]. sinsor2:[time1, time2, time3..]., ..}, id2}
	def create_detector_dictionary(self):
		print "Sorting Data by Ids..."
		rawData = self.raw_data
		i = 0
		for sensor in rawData.keys():
			print "      sorting id for sensor", sensor
			for id_ in rawData[sensor].keys():
				for time in rawData[sensor][id_]:
					if id_ not in self.detector_time.keys():
						self.detector_time[id_] = {}
					self.detector_time[id_][(str(i), time)] = str(int(sensor))
					i = i + 1

	def save_detector_dictionary(self,name):
		if len(name) > 7:
			self.dat_object.save_dict(self.detector_time, name, ';')
		print "Saved Sorted Trips as", name

	def load_trips(self, filename):
		if self.dat_object == "None":
			self.dat_object = FileManage(".csv", self.path)
		id_data = {}
		with open(filename, 'rb') as f:
			reader_ = csv.reader(f, delimiter=';')
			id_times = list(reader_)
			for data in id_times:
				time_list = ast.literal_eval(data[1])
				#time_id = [[time,x[:-4]] for time in time_list]
				id_data[data[0]]= time_list
		self.detector_time = id_data

		
	def get_trips(self):
		#for key in self.trips:
			#print key, self.trips[key]
		print "Total trips:", len(self.trips)
		#for key in self.valid_connections:
			#print key, self.valid_connections[key]
		print "Total Valid Trips:", len(self.valid_connections)


	def get_detector_time(self):
		print self.detector_time


	#{(id, start-time of trip):{(time,node_index):senosr....},...}
	def split_trips(self):
		print "Splitting trips from travel times..."
		all_data = self.detector_time
		trip_number = 0
		for _id in all_data:
			trip = {}
			id_info = all_data[_id]
			index_time = sorted(id_info, key = lambda x: x[-1])  
			i = 0
			index = 0

			tripmin = index_time[0][1]
			while i < len(index_time)-1:
				trip[(index_time[i][1], str(index))] = id_info[index_time[i]]
				index = index + 1
				(index1, t1) = index_time[i]
				(index2, t2) = index_time[i+1]
				time1 = datetime.utcfromtimestamp(t1)
				time2 = datetime.utcfromtimestamp(t2)
				diff = time2 - time1

				
				if diff.total_seconds() >= trip_differentiator*60:
					self.trips[(_id,tripmin)] = trip
					tripmin = index_time[i+1][1]
					trip = {} 
					index = 0


				i = i+1

			trip[(index_time[len(index_time)-1][1], str(index))] = id_info[index_time[len(index_time)-1]]
			self.trips[(_id,tripmin)] = trip

	def check_trips(self, kml):
		print "Checking Order of Sensors in Trips..."
		self.graph = CreateGraph()
		self.graph.graph_from_file(kml)
		self.graph.create_graph()
		for ids in self.trips:
			if len(self.trips[ids]) > 1:
				checks = []
				times = sorted(self.trips[ids])
				i = 0
				while i < len(times)-1:
					sensor1 = self.trips[ids][times[i]]
					sensor2 = self.trips[ids][times[i+1]]
					check = self.graph.check_order(sensor1,sensor2)
					checks.append(check)
					i = i+1

				if all(x==checks[0] for x in checks) == True:
					if checks[0] == True:
						self.valid_connections[ids] = self.trips[ids]

	def save_sorted_trips(self, name):
		if len(name) > 7:
			self.dat_object.save_dict(self.valid_connections, name, ';')
		print "Saved Sorted Trips as", name + ".csv"

	def load_sorted_trips(self, filename):
		if self.dat_object == "None":
			self.dat_object = FileManage(".csv", self.path)
		id_data = {}
		with open(filename, 'rb') as f:
			reader_ = csv.reader(f, delimiter=';')
			id_times = list(reader_)
			for data in id_times:
				time_list = ast.literal_eval(data[1])
				#time_id = [[time,x[:-4]] for time in time_list]
				id_data[data[0]]= time_list
		self.valid_connections = id_data

	def check_sensor(self,sensor,check):
		if type(sensor) == tuple or type(sensor)== list:
			for item in sensor:
				if item == check:
					return True
			return False
		else:
			if sensor == check:
				return True
			return False

	def check_sequence(self,trip,seq):
		index = None
		i = 0
		iseq = 0
		diff = len(trip)-len(seq)
		while i <= diff:
			if self.check_sensor(seq[0],trip[i][1]) == True:	
				index = i
				iseq = iseq + 1
				i = i + 1
				true = 1
				search = len(seq) + index
	 			while i < search:
				#for time,sensor in trip[i:i+len(seq)]:
					if self.check_sensor(seq[iseq],trip[i][1]) == False:
						i = i-1
						break
					else:
						true = true + 1
						iseq = iseq + 1
						i = i + 1
					if true == len(seq):
						return (True, trip[index:i])
			iseq = 0
			i = i + 1
			
		return (False,None)

	# def check_sequence(self,trip,seq):
	# 	index = None
	# 	i = 0
	# 	iseq = 0
	# 	diff = len(trip)-len(seq)
	# 	while i <= diff:
	# 		if self.check_sensor(seq[iseq],trip[i][1]) == True:	
	# 			index = i
	# 			iseq = iseq + 1
	# 			i = i + 1
	# 			true = 1
	# 			search = len(seq) - index
	# 			while i < search:
	# 			#for time,sensor in trip[i:i+len(seq)]:
	# 				if true == len(seq):
	# 					return (True, trip[index:i]) 
	# 				if self.check_sensor(seq[iseq],trip[i][1]) == False:
	# 					break
	# 				else:
	# 					true = true + 1
	# 					#if type(seq[iseq]) == tuple or type(seq[iseq]) == list:
	# 						#if self.check_sensor(seq[iseq+1],trip[i+1][1]) == True:
	# 							#i = i + 1
	# 				iseq = iseq + 1
	# 				i = i + 1
	# 		iseq = 0
	# 		i = i + 1
			
	# 	return (False,None)

	def sensor_in_trip(self, sensors):
		trips = {}
		first_sensor = sensors[0]
		trip_num = 1
		for id_, trip in self.valid_connections.items():
			for signal in trip.values():
				if self.check_sensor(first_sensor, signal) == True:
					sorted_trip = sorted(trip.items())
					res = self.check_sequence(sorted_trip,sensors)
					if res[0] == True:
						trip_info = res[1]
						trips[("Trip " + str(trip_num),id_,trip_info[0][0])] = trip_info
						trip_num = trip_num + 1
		return trips


rdm = RawDataManager('/Users/allibehr/Desktop/cmr/DataProcess')
#rdm.data_csv()
#rdm.create_detector_dictionary()
#rdm.save_detector_dictionary("Detectors") 
#rdm.get_detector_time()
#rdm.load_trips("Detectors.csv")
#rdm.split_trips()
#rdm.check_trips("SRIB.kml")
#rdm.get_trips()
rdm.load_sorted_trips('SortedTrips.csv')
#res = rdm.sensor_in_trip(['139','140',('128','130'),('120','129'),'119',('118','117')])
res = rdm.sensor_in_trip(['128','130'])
print res
print len(res)
