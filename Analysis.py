from datetime import datetime
from Data_Process import RawDataManager
import numpy as np
import csv
import os
import math
import random
from operator import itemgetter, attrgetter
from xlrd import open_workbook, xldate_as_tuple
import Filter
from Cluster import Cluster
from xlwt import Workbook, XFStyle

class DataAnalysis:

	def __init__(self, path_):
		self.path = path_
		self.data_object = None
		self.data = None
		self.forward_trips = {}
		self.backwards_trips = {}
		self.sort_ft = []
		self.sort_bt = []
		self.forward_split = {}
		self.backward_split = {}
		self.outliers = {}
		self.c_object = None
		self.compiled_data = {}
		self.sensor_times = {}
		self.total_time = {}

	def count_items(self, dict_):
		num = 0 
		for items in dict_:
			num += len(dict_[items])
		return num

	def load_data(self,file_):
		"""Loads sorted valid trips from DataProcess class"""
		self.data_object = RawDataManager(self.path)
		self.data_object.load_sorted_trips(file_)
		self.data = self.data_object.get_valid_trips()

	def get_trips(self,sequence):
		"""finds trips in all valid trips that contain the desired 
		sequence of sensors where sequence = [sensor1, sensor2....]"""
		self.forward_trips = self.data_object.check_seq(sequence)
		print "Foward Trips:", len(self.forward_trips)
		reverse = sequence
		reverse.reverse()
		self.backwards_trips = self.data_object.check_seq(reverse)
		print "Backward Trips:", len(self.backwards_trips)

	def find_times(self, timelist):
		"""finds the start time and end time in a list of sensor data, 
		where some sensors may have two readings. In this case the first sensor is taken"""
		if type(timelist[0][1]) == tuple:
			start_time = sorted(timelist[0], key = lambda x: x[-1])[0][0][0]
		else:
			start_time = timelist[0][0][0]
		if type(timelist[-1][1]) == tuple:
			end_time = sorted(timelist[-1], key = lambda x: x[-1])[0][0][0]
		else:
			end_time = timelist[-1][0][0]

		return (datetime.utcfromtimestamp(start_time),datetime.utcfromtimestamp(end_time))

	def list_times(self, group, type0_ = None):
		"""for a group of 50 trips, this finds the start time, end time and total time for each trip"""
		start_times = []
		end_times = []
		for trip in group:
			(start, end) = self.find_times(trip[1])
			if type0_ == 'time':
				start = start.time()
				end = end.time()

			start_times.append(start)
			end_times.append(end)

		return start_times, end_times

	def diff_time(self, time1, time2):
		"""finds the difference in minutes between two timestamps"""
		start = datetime.utcfromtimestamp(time1)
		end = datetime.utcfromtimestamp(time2)
		total_time = end - start
		return total_time.total_seconds()/60.

	def sort_trips(self,type_ = None):
		"""sorts trips containing desired sequnce by day and time or just time"""
		if type_ == 'day' or type_ == None:
			self.sort_ft = sorted(self.forward_trips.items(), key = lambda x: x[0][2])
			self.sort_bt = sorted(self.backwards_trips.items(), key = lambda x: x[0][2])

		if type_ == 'time':
			self.sort_ft = sorted(self.forward_trips.items(), key = lambda x: datetime.utcfromtimestamp(x[0][2]).time())
			self.sort_bt = sorted(self.backwards_trips.items(), key = lambda x: datetime.utcfromtimestamp(x[0][2]).time())

	def split_trips(self, group_number, overlap = None):
		"""splits sorted trips into groups of fifty, if overlap it set to a number, groups will overlap by that number"""
		tripi = 0
		group = 1
		while tripi < len(self.sort_ft):
			i = 0
			trips = []
			while i < group_number:
				if tripi < len(self.sort_ft):
					trips.append(self.sort_ft[tripi]) 
					tripi = tripi + 1
					i = i+1
				else:
					break

			if len(trips) < 10:
				old_trips = self.forward_split['Group ' + str(group-1)]
				new_trips = old_trips + trips
				self.forward_split['Group ' + str(group-1)] = new_trips
				break
				
			self.forward_split['Group ' + str(group)] = trips
			group = group + 1
			if overlap:
				if tripi < len(self.sort_ft):
					if overlap < group_number:
						tripi = tripi - overlap
					else:
						raise ValueError, 'overlap value too large'
		
		tripi = 0
		group = 1
		while tripi < len(self.sort_bt):
			i = 0
			trips = []
			while i < group_number:
				if tripi < len(self.sort_bt):
					trips.append(self.sort_bt[tripi]) 
					tripi = tripi + 1
					i = i+1
				else:
					break

			if len(trips) < 10:
				old_trips = self.forward_split['Group ' + str(group-1)]
				new_trips = old_trips + trips
				self.backward_split['Group ' + str(group-1)] = new_trips
				break

			self.backward_split['Group ' + str(group)] = trips
			group = group + 1
			if overlap:
				if tripi < len(self.sort_bt):
					if overlap < group_number:
						tripi = tripi - overlap
					else:
						raise ValueError, 'overlap value too large'

	def get_avg_stdv(self, timelist):
		"""returns average and stdv of a list of values (timelist)"""
		sum_ = 0
		var = 0.
		for times in timelist:
			sum_ = sum_ + times
		average = sum_/len(timelist)

		for times in timelist:
			var = var + (times-average)**2
		var = var/len(timelist)
		std = np.sqrt(var)

		return average, std

	def get_time_value(self,info):
		"""returns time of sensor detection from sensor list"""
		if type(info[1]) == tuple:
			time = sorted(info, key = lambda x: x[-1])[0][0][0]
		else:
			time = info[0][0]
		return time

	def find_R(self,X, Y):
		"""finds the R value from two lists of link times"""

		if len(X) != len(Y):
			print X, Y
			raise ValueError, 'unequal length' 

		sum_ = error = residual = 0.

		avgx, stdvx = self.get_avg_stdv(X)
		avgy, stdvy = self.get_avg_stdv(Y)

		n = float(len(X))

		for x, y in zip(X, Y):
			xin = (x-avgx)/stdvx
			yin = (y-avgy)/stdvy
			sum_ = sum_ + xin*yin

		R = sum_/(n-1)
		return R

	def find_sensor(self, tuple_):
		"""used to find time traveled on a link, this finds the sensor_name in group sensor data"""
		if type(tuple_[1]) == tuple:
			sensors = []
			for tuples in tuple_:
				sensors.append(tuples[1])
			return sorted(sensors)[0]
		return tuple_[1]

	def split_sensors(self, list_):
		"""splits the time data in a group of 50 into sensors instead of trips"""
		sensors = {}
		keys = []
		for signal in list_[0][1]:
			sensors[self.find_sensor(signal)] = []
			keys.append(self.find_sensor(signal))

		for trip in list_:
			i = 0
			for signal in trip[1]:
				long_time = self.get_time_value(signal)
				sensors[keys[i]].append(long_time)
				i = i + 1

		return sensors, keys

	def time_between(self,sensor_list):
		"""finds the time between two sensors"""
		time_values = []
		sensors, keys = sensor_list
		for i in xrange(len(keys)-1):
			times = []
			link = keys[i] + '_' + keys[i+1]
			sensor1 = sensors[keys[i]]
			sensor2 = sensors[keys[i+1]]

			for i in xrange(len(sensor1)):
				time = self.diff_time(sensor1[i],sensor2[i])
				times.append(time)
			time_values.append((link,times))

		return time_values

	def get_trip_times(self, time_list):
		trips = {}
		for i in range(len(time_list[0])):
			times = [link[i] for link in time_list]
			trips[i+1] = times
		return trips

	def find_totals(self, sensor_list):
		all_totals = []
		#print sensor_list
		for j in range(len(sensor_list[0][1])):
			total = 0
			for i in range(len(sensor_list)):
		#		print sensor_list[i][j]
				total += sensor_list[i][1][j]
			all_totals.append(total)
		return all_totals

	def filter_data(self, sensors, group):
		sensor_data = [sensor_dt[0] for sensor_dt in sensors]
		times = [sensor_dt[1] for sensor_dt in sensors]
		info = self.get_trip_times(times)
	#	print "Info", info
		fi = Filter.Final(info)
		trips_new, self.outliers[group] = fi.get_new_trips()
		#print "New", trips_new
		#print len(trips_new)
		print "removed: ", len(self.outliers[group])
		temp = []
		for sensor in sensor_data:
			temp.append((sensor, []))
		#print temp
		for trip in sorted(trips_new.keys()):
			for i in range(len(trips_new[trip])):
				temp[i][1].append(trips_new[trip][i])
		return temp

	def get_excel_data(self):
		new_data = {}
		sensor_times = {}
		total_time = {}

		book = open_workbook("I-5_S_BlueStats 39-9-10-11.xls")
		sheet = book.sheet_by_name('Data-1')
		mac_ids = sheet.col_values(0,start_rowx=1)
		starttime = sheet.col_values(4,start_rowx=1)
		tt_39_09 = sheet.col_values(7,start_rowx=1)
		tt_09_10 = sheet.col_values(11,start_rowx=1)
		tt_10_11 = sheet.col_values(15,start_rowx=1)
		totals = sheet.col_values(17,start_rowx=1)
		go = True
		last = False 
		start = 0
		end = 50
		g_index = 0

		while go:
			if last:
				go = False

			g_index += 1
			group = "Group " + str(g_index)
			stimes = starttime[start:end]
			t1 = tt_39_09[start:end]
			t2 = tt_09_10[start:end]
			t3 = tt_10_11[start:end]
			total = totals[start:end]
			sensors = [('39_09', t1),('09_10', t2),('10_11', t3)]
			sorted_sensors = self.filter_data(sensors,group)

			Rvalues = []
			j = 0
			while j < len(sorted_sensors)-1:
					(link1,times1) = sorted_sensors[j]
					(link2,times2) = sorted_sensors[j+1]
					link = 'tt_'+link1+'/' + link2
					#print link
					R =self.find_R(times1,times2)
					Rvalues.append((link,R))
					j = j + 1

			new_data[group] = [r[1] for r in Rvalues]
			sensor_times[group] = sorted_sensors
			total_time[group] = total  
			start = end
			end += 50

			if end >= len(mac_ids):
				end = len(mac_ids)
				last = True

			g_index += 1
		return new_data, None, sensor_times, total_time


	def compile_data(self,dict_, type_ = None, filter_ = True):
		"""compiles all data ("R" value, Timespan of group (first starttime: laststarttime)),
		 the average time and stdv for each link, the average trip time, trip stdv) and 
		returns in dictionary. Also returns sensor times and total times"""
		data = {}
		sensor_times = {}
		total_time = {}
		for group in dict_:
			Rvalues = []
			j = 0
			sensorlist = self.split_sensors(dict_[group])
			sensors = self.time_between(sensorlist)
			#print "sensors" , sensors
			if filter_:
				sensors = self.filter_data(sensors, group)

			sensor_times[group] = sensors
			totals = self.find_totals(sensors)
			total_time[group] = totals

			while j < len(sensors)-1:
				(link1,times1) = sensors[j]
				(link2,times2) = sensors[j+1]
				link = 'tt_'+link1+'/' + link2
				#print link
				R =self.find_R(times1,times2)
				Rvalues.append((link,R))
				j = j + 1

			start, end = self.list_times(dict_[group], type_)
			avgtimes = []
			for sensor in sensors:
				avg, stdv = self.get_avg_stdv(sensor[1])
				avgtimes.append((sensor[0],avg,stdv))

			timespan = self.timespan(start)
			average , stdv = self.get_avg_stdv(totals)
			
			#, 'Start,End,Total(min)':times, 
			data[group] = {'R':Rvalues, 'Timespan':timespan, 'Segment:(Average time(sec), Average stdv)': avgtimes, 'Average_trip(min)':average, 'Stdv(min)':stdv}
		print "points removed:", self.count_items(self.outliers)
		self.compiled_data = data
		self.sensor_times = sensor_times
		self.total_time = total_time
		return data, sensor_times, total_time

	def get_group_rvalues(self, sensors, type_ = 'time', reverse = False):
		long_, longtime, l_times = self.get_long(sensors, type_ = type_, reverse = reverse)
		GR = {}
		for group in long_:
			rlist = long_[group]['R']
			rs = [r[1] for r in rlist]
			#print rs
			GR[group] = rs

		return GR, long_, longtime, l_times

	def cluster_data(self, sensors, type_ = 'time', reverse = False):
		GR, long_, longtime, l_times = self.get_group_rvalues(sensors, type_ = None, reverse = reverse)
		#GR, long_, longtime, l_times = self.get_excel_data()
		self.c_object = Cluster(GR)
		gc = self.c_object.main()
		total_times = {}
		link_times = {}
		for cluster in gc:
			link_times[cluster] = {}
			total_times[cluster] = []
			for i in range(len(longtime[longtime.keys()[0]])):
				link_times[cluster][i+1] = []
			for group in gc[cluster]:
				for i in range(len(longtime[group])):
					ti = longtime[group][i][1]
					link_times[cluster][i+1].extend(ti)
				total_times[cluster].extend(l_times[group])
			print "trips in cluster " + str(cluster), len(total_times[cluster])
		#print "c_object_1", c_object
		return gc, link_times, total_times

	def load_segment(self, sensors, type0_ = 'time'):
		"""loads a segment of grid"""
		self.load_data('SortedTrips.csv')
		self.get_trips(sensors)
		self.sort_trips(type_ = type0_)
		self.split_trips(20)
		foward, forward_time, tottimes_f = self.compile_data(self.forward_split,type_ = type0_)
		backward, backward_time, tottimes_b = self.compile_data(self.backward_split,type_ = type0_)
		return foward, backward, forward_time, backward_time, tottimes_f, tottimes_b

	def get_long(self, sensors, type_ = None, reverse = False):
		"""get the direction of segment with the most trips""" 
		forward, backward, forward_time, backward_time, tottimes_f, tottimes_b = self.load_segment(sensors, type0_ = type_)
		if len(forward) >= len(backward):
			long_ = forward
			longtime = forward_time
			l_times = tottimes_f
			short = backward
			shorttime = backward_time
			s_times = tottimes_b
		else:
			long_ = backward
			longtime = backward_time
			l_times = tottimes_b
			short = forward
			shorttime = forward_time
			s_times = tottimes_f

		if reverse:
			long_ = short
			longtime = shorttime
			l_times = s_times
		return long_, longtime, l_times

	"""finds begining and end of list of times"""	
	def timespan(self,times):
		s_times = sorted(times)
		return (s_times[0].isoformat(),s_times[-1].isoformat())

	def save_file(self, filename, dict_, del_):
		with open(filename + ".csv", 'wb') as f:
		    writer = csv.writer(f)
		    writer = csv.writer(f, delimiter=del_)
		    for row in dict_.iteritems():
		        writer.writerow(row)	

	def save_R_values(self, dicts):
		with open('Rvalues.csv', 'wb') as f:
			for direction in dicts:
				for group in direction.keys():
					writer = csv.writer(f)
					writer.writerow('')
					writer.writerow(direction[group]['Start,End,Total(min)'])
					fieldnames = direction[group]['R'].keys()
					writer = csv.DictWriter(f, fieldnames=fieldnames)
					writer.writeheader()
					writer.writerow(direction[group]['R'])

	def main(self, streets, type_ = 'time'):
		"""returns data from a dictionary containing streets: {segmentname:[sensors...], ....}"""
		for street in streets:
			da = DataAnalysis(pathname)
			forward, backward, foward_time, backward_time, tottimes_f, tottimes_b = da.load_segment(streets[street], type0_ = type_)
			if len(forward) >= len(backward):
				longest = forward
			else:
				longest = backward
			print street, len(longest)
			da.save_file(street, longest, ';')





#streets = {"Centre":['106','107','113',('112','115')], "Highland":[('120','129'),'136','135','134'], 'Penn':['139', '140', ('128', '130'), ('120', '129'), '119'],"Baum":['104',('103','109'),'102']}
# links = {'PennOne':['139','140',('128','130')], 'PennTwo':['140',('128','130'),('120', '129')], 'Three':['116',('115','112'),'113']}
#da = DataAnalysis(pathname)
#da.main(streets)
# forward, backward, foward_time, backward_time = da.load_segment([('109','103'), '110', '140'], type0_ = 'time')
# da.save_file('Beatty_Forward', forward, ';')
# da.save_file('Beatty_Backward', backward, ';')




