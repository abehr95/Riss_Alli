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

trip_differentiator = 10
max_rep = 3
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
		self.number_trips = None
		self.valid_trips = None
		self.invalid_trips = None
		self.repeats = 0
		self.repeat_rmv = []


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
		dels = []
		for key in self.detector_time:
			if all(x==self.detector_time[key].values()[0] for x in self.detector_time[key].values()) == True:
				dels.append(key)
				self.repeats += 1

		for item in dels:
			del self.detector_time[item]

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

		
	def get_valid_trips(self):
		return self.valid_connections

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
					trip_number += 1


				i = i+1

			trip[(index_time[len(index_time)-1][1], str(index))] = id_info[index_time[len(index_time)-1]]
			self.trips[(_id,tripmin)] = trip

		self.number_trips = trip_number

	def check_trips(self, kml):
		print "Checking Order of Sensors in Trips..."
		vt = 0
		it = 0
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

				num_rest = 0
				for res in checks:
					if res == 'Resting':
						num_rest += 1
				if num_rest > 0:
					j = 0
					while j < num_rest-1:
						checks.remove('Resting')
						j += 1
					self.repeat_rmv.append(num_rest)

				if all(x==checks[0] for x in checks) == True:
					if checks[0] == True:
						self.valid_connections[ids] = self.trips[ids]
						vt += 1
				else:
					it += 1

		self.valid_trips = vt
		self.invalid_trips = it

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

	def check_element(self, check1, check2, expect, i):
		if type(expect) == tuple:
			res = None
			if check1[1] == expect[0] or check1[1] == expect[1]:
				res = check1
				if check2[1] == expect[0] or check2[1] == expect[1]:
					res = (check1,check2)
					i = i + 1
				return True, i + 1, res
			return False,  i + 1, None

		else:
			if check1[1] == expect:
				return True,  i + 1, check1
			return False, i + 1, None

	def check_seq(self, seq):
		trips = {}
		trip_num = 1
		tups = 0
		for id_, trip in self.valid_connections.items():
			s_trip = sorted(trip.items())
			sorts = []
			i1 = 0
			length = len(s_trip)-len(seq)
			if len(s_trip) >= len(seq):
				while i1 <= length:
					for sensor in seq:
						if i1 < len(s_trip)-1:
							check = self.check_element(s_trip[i1], s_trip[i1+1], sensor, i1)
							if check[0] == True:
								i1 = check[1]
								sorts.append(check[2])
							else:
								break
							if len(sorts) == len(seq):
								trips[('Trip ' + str(trip_num),ast.literal_eval(id_)[0],self.start_time(sorts))] = sorts
								trip_num = trip_num + 1
								sorts = []
						if i1 == len(s_trip)-1:
							check = self.check_element(s_trip[i1], 'None', sensor, i1)
							if check[0] == True:
								sorts.append(check[2])
							else:
								break
							if len(sorts) == len(seq):
								trips[('Trip ' + str(trip_num),ast.literal_eval(id_)[0],self.start_time(sorts))] = sorts
								trip_num = trip_num + 1
								sorts = []
					i1 = i1 + 1
					sorts = []	
		return trips

	def start_time(self,timelist):
		if type(timelist[0][1]) == tuple:
			start_time = sorted(timelist[0], key = lambda x: x[-1])[0][0][0]
		else:
			start_time = timelist[0][0][0]
		return start_time

	def sort_trips_all(self):
		self.data_csv()
		self.create_detector_dictionary()
		self.save_detector_dictionary('Detectors')
		self.split_trips()
		self.check_trips('SRIB.kml')
		self.save_sorted_trips('SortedTrips')
		self.check_stats()

	def get_num_detectors(self):
		return len(self.detector_time)

	def get_num_trips(self):
		return self.number_trips

	def get_check_valid(self):
		return self.valid_trips

	def get_check_invalid(self):
		return self.invalid_trips

	def get_removed_repeats(self):
		return self.repeat_rmv

	def get_repeats(self):
		return self.repeats

	def check_stats(self):
		print "Number of MacIDs: ", self.get_num_detectors()
		print "After Split... Number of Trips: ", self.get_num_trips()
		print "After Check... Valid Trips: ", self.get_check_valid(), " Invalid Trips: ", self.get_check_invalid()
		print "Repeats removed from sequence: ", len(self.get_removed_repeats())
		print "Trips Discard due to repeats over max: ", self.get_repeats()
		print "Differentiator: ", trip_differentiator
		print "Repeat Threshold: ", max_rep


#rdm = RawDataManager('/Users/allibehr/Desktop/cmr/DataProcess')
#rdm.data_csv()
#rdm.create_detector_dictionary()
#rdm.save_detector_dictionary('Detectors')
#rdm.split_trips()
#rdm.check_trips('SRIB.kml')
#rdm.save_sorted_trips('SortedTrips')
# rdm.load_sorted_trips('SortedTrips.csv')
# print len(rdm.valid_connections)
# res = rdm.check_seq(['139', '140', ('128', '130'), ('120', '129'), '119', ('118', '117')])
# #res = rdm.sensor_in_trip(['140',('128','130')])
# #rdm.check_seq_penn()
# #res = rdm.check_seq_penn()
# print res
# print len(res)

#import Data_Process
#rdm = Data_Process.RawDataManager('/Users/allibehr/Desktop/cmr/DataProcess')
#rdm.sort_trips_all()

