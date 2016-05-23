import numpy as np
import json
import sys
import csv
from os import walk
import ast
from fastkml import kml
from shapely.geometry import Point, LineString, Polygon

trip_differentiator = 100
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
				if 'trips' not in el:
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
				if self.is_empty(times) == False:
					self.file_data[float(file_[:-4])] = times
		elif self.file_type == ".csv":
			for file_ in files:
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
			with open(sensor + ".csv", 'wb') as f:
			    writer = csv.writer(f)
			    writer = csv.writer(f, delimiter=';')
			    for row in id_data.iteritems():
			        writer.writerow(row)

	def save_sensor(self,sensor):
		sensor_name = str(sensor)
		if sensor_name in self.file_data.keys():
			id_data = self.file_data[sensor]
			with open(sensor + ".csv", 'wb') as f:
			    writer = csv.writer(f)
			    writer = csv.writer(f, delimiter=';')
			    for row in id_data.iteritems():
			        writer.writerow(row)
		else:
			raise ValueError('Cannot save file: ' + sensor_name + '. It does not exist')

	def get_all_files(self):
		return self.all_files


class RawDataManager:

	def __init__(self, path_):
		#path is file path where data is stored

		#trips will contain id:{sensor:[times]}, 

		#rawdata will be filled according to {sensor:{ida:[time1, time2,...], idb:[time1, time2, time3....], ...}, sensor 2:{}}

		self.trips = {}
		self.path = path_
		self.raw_data = {}
		dat_object = "None"

	##uses FileManager class to retrieve sensor data 
	#from dat file to fill rawdata. save all files into csv form.
	def data_dat(self):
		dat_object = FileManage(".dat", self.path)
		dat_object.load_all_files()
		self.raw_data = dat_object.get_data()
		dat_object.save_all()

	#gets data to sill raw_data object from csv file. 
	def data_csv(self):
		dat_object = FileManage(".csv", self.path)
		dat_object.load_all_files()
		self.raw_data = dat_object.get_data()

	#saves single sensor files
	def save_sensor(sensor_):
		dat_object.save_sensor(sensor_)

	#trips data is created by sorting data into {id1:{sensor1:[time1, time2...]. sinsor2:[time1, time2, time3..]., ..}, id2}
	def create_detector_dictionary(self):
		rawData = self.raw_data
		for sensor in rawData.keys():
			for id_ in rawData[sensor].keys():
				if id_ not in self.trips.keys():
					self.trips[id_] = {}
				self.trips[id_][sensor] = rawData[sensor][id_]

	def get_trips():
		return self.trips


	#NOT PUT INTO CLASSES DEF. How do i handle list of times. how do i handle determining valid trip???
	def sort_all_data(data):
		'''
		ids = {}
		for detector in data.keys():
			for id_ in detector[1].items():
				if ids.has_key(id_[0]):
					ids[id_[0]]=ids[id_0].append({detector[0]:id_[1]})
				else:
					ids[id_[0]]=[{detector[0]:id_[1]}]
		return ids
		'''

	def split_trips(self):
		'''
		all_data = create_detector_dictionary()
		trip_number = 1
		trip_dictionary = {}
		for id_ in all_data.keys():
			times = all_data[id_]
			i = 1
			trip = []
			while i < len(times):
				#print times[i] , times[i-1] , id_, 
				time_between = times[i][0]-times[i-1][0]
				trip.append(times[i-1])
				if time_between >= trip_differentiator:
					trip_dictionary['trip_' + str(trip_number)] = [id_, trip]
					trip_number = trip_number + 1
					trip = []
				i = i+1
		return trip_dictionary
		'''

class Map:

	def __init__(self, path_):
		kml_object = kml.KML()
		self.path = path_

	def kml_to_string(self):
		with open(self.path, 'r') as myfile:
			mapInfo=myfile.read()	
		return mapInfo
	
	def create_kml_object(self):
		self.kml_object.from_string(self.kml_to_string())

	def list_features(self):
		print list(self.kml_object.features())





#graph = {}

# def save_detectors():
# 	for detector in get_files('.dat'):
# 		save(sort_data(load_dat(detector)),detector)

# def save_trips():
# 	save(split_trips(),'trips.csv')

# def get_coordinates():
# 	filenames = get_files('.kml')
# 	for filename in filenames:
# 		from pykml import parser
# 		root = parser.fromstring(open(filename, 'r').read())
# 		print root.Document.Placemark.Point.coordinates

# def main():
# 	save_detectors()
# 	save_trips()
# 	get_coordinates()

# main()