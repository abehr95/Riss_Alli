from datetime import datetime
from Data_Process import RawDataManager
import numpy as np
import csv
import os
import math
import random
from operator import itemgetter, attrgetter
import Filter

pathname = os.getcwd()

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
		"""finds the difference in seconds between two timestamps"""
		start = datetime.utcfromtimestamp(time1)
		end = datetime.utcfromtimestamp(time2)
		total_time = end - start
		return total_time.total_seconds()

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
			times = [link[i] for link in time_dict]
			trips[i+1] = times
		return trips

	def find_totals(self, sensor_list):
		all_totals = []
		for j in range(len(sensor_list[0])):
			total = 0
			for i in range(len(sensor_list)):
				total += sensor_list[i][j]
			all_totals.append(total)
		return all_totals

	def compile_data(self,dict_, type_ = None, sort = True):
		"""compiles all data ("R" value, Timespan of group (first starttime: laststarttime)),
		 the average time and stdv for each link, the average trip time, trip stdv) and 
		returns in dictionary. Also returns sensor times and total times"""
		data = {}
		sensor_times = {}
		total_time = {}
		for group in dict_:
			Rvalues = {}
			i = 0
			sensorlist = self.split_sensors(dict_[group])
			sensors = self.time_between(sensorlist)
			if sort_:
				sensor_data = [sen[0] for sen in sensors]
				times = [time[1] for times in sensors]
				info = self.get_trip_times(times)
				print "Info", info
				fi = Filter.Final(info)
				trips_new, self.outliers[group] = fi.get_new_trips()
				#print "New", trips_new
				#print len(trips_new)
				temp = []
				for trip in sorted(trips_new.keys()):
					for i in range(len(trips_new[trip])):
						temp.append((sensor_data[i], trips_new[trip][i]))
				print temp
				print "Trips Removed from " + group +":", len(self.outliers)
				sensors = temp

			sensor_times[group] = sensors
			totals = self.find_totals(sensors)
			total_time[group] = totals

			while i < len(sensors)-1:
				(link1,times1) = sensors[i]
				(link2,times2) = sensors[i+1]
				link = 'tt_'+link1+'/' + link2
				#print link
				Rvalues[link] = self.find_R(times1,times2)
				i = i + 1

			start, end = self.list_times(dict_[group], type_)
			avgtimes = []
			for sensor in sensors:
				avg, stdv = self.get_avg_stdv(sensor[1])
				avgtimes.append((sensor[0],avg,stdv))

			timespan = self.timespan(start)
			average , stdv = self.get_avg_stdv(totals)
			
			#, 'Start,End,Total(min)':times, 
			data[group] = {'R':Rvalues, 'Timespan':timespan, 'Segment:(Average time(sec), Average stdv)': avgtimes, 'Average_trip(min)':average, 'Stdv(min)':stdv}

		return data, sensor_times, total_time

	def load_segment(self, sensors, type0_ = None):
		"""loads a segment of grid"""
		self.load_data('SortedTrips.csv')
		self.get_trips(sensors)
		self.sort_trips(type_ = type0_)
		self.split_trips(50, overlap = 25)
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

	def main(self, streets, type_ = None):
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


class Convolve:

	def __init__(self, segment_):
		self.analysis = DataAnalysis(pathname)
		self.segment = segment_
		self.times = {}
		self.links = []
		self.longtimes = {}
		self.data = {}
		self.actual_cdfs = {}
		self.link_cdfs = {}
		self.comon_cdfs = {}
		self.convo_cdfs = {}

	def get_times_data(self, type_, sort_ = True):
		print "Accquiring data..."
		self.data, times, self.longtimes = self.analysis.get_long(self.segment, type_ = type_)
		self.RValues = self.data['R']
		for group in times:
			self.times[group] = {}
			for i in range(len(times[group])):
				sensor_dt = times[group][i]
				sensor = sensor_dt[0]
				times_ = sensor_dt[1]
				self.times[group][sensor] = times_

		for i in range(len(times[times.keys()[0]])):
			self.links.append(times[times.keys()[0]][i][0])

		if sort_:
			temp = {}
			for group in self.times:
			#if len(self.times[group][self.times[group].keys()[0]]) > 10:
				temp[group] = {}
				info = self.times[group]
				#print len(self.get_trip_times(info))
				#print "Info", info
				#print "Orig", self.get_trip_times(info)
				fi = Filter.Final(self.get_trip_times(info))
				trips_new, removed = fi.get_new_trips()
				#print "New", trips_new
				#print len(trips_new)
				for link in self.links:
					temp[group][link] = []
				for trip in sorted(trips_new.keys()):
					for i in range(len(trips_new[trip])):
						temp[group][self.links[i]].append(trips_new[trip][i])
				print "Trips Removed from " + group +":", removed
			self.times = temp

	def get_trip_times(self, time_dict):
		trips = {}
		for i in range(len(time_dict[time_dict.keys()[0]])):
			times = [time_dict[link][i] for link in time_dict.keys()]
			trips[i+1] = times
		return trips

	def actual_time_cdf(self):
		print "Finding actual CDFs..."
		cdfs = {}
		for group in self.times:
			times = []
			for i in range(len(self.times[group][self.times[group].keys()[0]])):
				time = 0
				for link in self.times[group]:
					time += self.times[group][link][i]
				times.append(time)
			cdf = self.compute_cdf(times)
			cdfs[group] = cdf
		self.actual_cdfs = cdfs
	
	def count_record(self,dict_type):
		L = sorted(dict_type.viewkeys(), reverse = False)
		n = 0
		for i in xrange(len(L)):
			z = L[i]
			temp_1 = dict_type[z]
			n += temp_1
		return n

	def percentile(self, N, percent):
		"""
		Find the percentile of a list of values.

		@parameter N - is a list of values. Note N MUST BE already sorted.
		@parameter percent - a float value from 0.0 to 1.0.
		@return - the percentile of the values
		"""
	
		if not N:
			return None
		k = (len(N)-1) * percent
		floor = math.floor(k)
		ceil = math.ceil(k)
		if floor == ceil:
			return (N[int(k)])
		val_1 = (N[int(floor)]) * (ceil-k)
		val_2 = (N[int(ceil)]) * (k-floor)
		return val_1+val_2

	def compute_cdf(self, _list_travel_times):
		_cdf = {}
		sorted_travel_times = sorted(_list_travel_times)
		for percent in xrange(0, 101):
			percent = round(float(percent/100.), 2)
			percent_tt = self.percentile(sorted_travel_times, percent)
			_cdf[percent] = percent_tt
		return _cdf

	def get_link_cdfs(self):
		print "Finding link CDFs..."
		for group in self.times:
			self.link_cdfs[group] = {}
			for link in self.times[group]:
				cdf = self.compute_cdf(self.times[group][link])
				self.link_cdfs[group][link] = cdf

	def route_cdf_comonotonic(self):
		"""This method computes synthesized route CDF under the assumption adjacent
		segments have strong positive dependencies or in other words, comonotonic"""
		print "Finding CDFs using comonotoncity..."
		comonotonic_cdfs = {}

		for group in self.link_cdfs:
			comonotonic_cdfs[group] = {}
			routes = self.link_cdfs[group].keys()

			percents = sorted(self.link_cdfs[group][routes[0]].viewkeys(), reverse = False)

			for i in range(len(percents)):
				perc = percents[i]
				route_tt = 0
				for link in self.link_cdfs[group]:
					route_tt += self.link_cdfs[group][link][perc]
				comonotonic_cdfs[group][perc]= route_tt

		self.comon_cdfs = comonotonic_cdfs

	def dictionary_update(self,dict_type, key):
		"""Takes user specified name of dictionary and key as input argument and
		generates dictionary (key = Individual vehicle travel rate generataed
		from each Monte Carlo run, value = frequency of that travel rate) for various
		operating conditions"""
		
		if key in dict_type:
			dict_type[key] += 1
		else:
			dict_type[key]= 1

	def percentile_freq(self, dict_type, perc_data):
		"""Takes dictionary (key = travel rate, and value = frequency) as input argument,
		and computes percentiles of CDF, and returns those results as a dictionary"""
		#list_keys = list(dict_type.viewkeys())
		#print len(list_keys)
		#print list_keys
		L = sorted(dict_type.viewkeys(), reverse = False )
		#print L
		n = self.count_record(dict_type)
		cum_freq = 0
		prev = 0
		k = 0.0
		prev_k = 0
		for j in xrange(len(L)):
			#print k
			perc = round(k*n,0)            
			temp = L[j]
			frequency = dict_type[temp]
			#print "frequency: " + str(frequency)
			cum_freq += frequency
			#print "cum_freq: " + str(cum_freq)
			prev = (cum_freq - frequency)
			#print "prev: " + str(prev)
			#test = round(cum_freq/n,4)

			while cum_freq >= perc:
				if k > 1:
					if 1 not in perc_data.keys():
						temp = perc_data[0.99]
						perc_data[1]= temp
					break
				val = round(k,2)
				perc_data[val]= temp
				k += 0.01
				perc = round(k*n,0)

		return perc_data

	def route_cdf_convolve(self):
		"""Independence between adjacent segments is assumed, therefore route CDF is
		computed by convolving individual link CDFs (monte carlo simulation is performed
		to achieve this)"""
		print "Finding CDFs using convolution..."
		convolve_cdfs = {}
		for group in self.link_cdfs:
			print group
			convolve_cdfs[group] = {}
			routes = self.link_cdfs[group].keys()
			temp = {}
			for i in range(100000):
				rt_tt = 0
				for link in self.link_cdfs[group]:
					r1 = round(random.uniform(0,1),2)
					tt_lnk = self.link_cdfs[group][link][r1]
					rt_tt += tt_lnk
				self.dictionary_update(temp, rt_tt)

			convolve_cdfs[group] = self.percentile_freq(temp, convolve_cdfs[group])

		self.convo_cdfs = convolve_cdfs


	def composite_cdf(self,alpha,data_type_1,data_type_2):
		"""This method takes alpha (weight factor for convolution) as an input parameter
		and computes composite CDF (by method of convolution) of synthesized route CDFs
		obtained by convolution(stratafied monte carlo sampling), and by adding percentiles"""
		#print "COMPOSITE CDF"
		temp = {}
		composite = {}
		n1 = int(round(10000*alpha,0))
		#print n1
		n2 = 10000 - n1
		for i in range(n1):
			r1 = round(random.uniform(0,1),2)
			#print r1
			#if r1 == 0:
			#	r1 = 0.01
			tt_1 = data_type_1[r1]
			self.dictionary_update(temp, tt_1)
		for j in range(n2):
			#print r2
			r2 = round(random.uniform(0,1),2)
			#if r2 == 0:
			#	r2 = 0.01 
			tt_2 = data_type_2[r2]
			self.dictionary_update(temp, tt_2)

		#print "Temp", temp
		composite = self.percentile_freq(temp, composite)
		#print "Composite", composite
		return composite

	def percentile_difference(self,dict_actual, dict_other):
		"""Takes CDF tables as input arguments and computes percentage error between
		individual percentile values"""
		freq_table = {}
		diff_table = {}
		m = sorted(dict_actual.keys(), reverse = False)
		square_error = 0
		for i in range(len(m)):
			key = m[i]
			perc_diff = 100*((dict_other[key]- dict_actual[key])/dict_actual[key])
			perc_error = round(perc_diff,2)
			square_error += round((dict_other[key]-dict_actual[key])**2, 3)
			diff_table[key]= perc_error
			self.dictionary_update(freq_table, perc_error)
		n = sorted(freq_table.keys(), reverse = False)
		tot = 0
		for j in range(len(n)):
			curr_key = n[j]
			if curr_key >= -1.5 and curr_key <= 1.5:
				tot += freq_table[curr_key]
		if tot >= 85:
			test = 1
		else:
			test = 0

		rmse = math.sqrt(square_error/100.0)
		return tot, test, freq_table, diff_table, rmse

	def test_significance(self):
		"""This method convolves synthesized CDFs sampling each CDF on a weighted
		percentage alpha(alpha varies between 0-1), and performs KS Test to check for
		statistical significance between actual_cdf and newly synthesized cdf"""
		print "Testing Significance..."
		all_data = {}
		for group in self.actual_cdfs:
			alpha = 0
			curr_dict_comon = self.comon_cdfs[group]
			curr_dict_conv = self.convo_cdfs[group]
			curr_dict_route = self.actual_cdfs[group]
			update_dict = {}
			print group
			s2, t2, freq_conv, diff_conv, rmse_2 = self.percentile_difference(curr_dict_route, curr_dict_conv)
				#print 's2: ' + str(s2)
			s3, t3, freq_comon, diff_comon, rmse_3 = self.percentile_difference(curr_dict_route, curr_dict_comon)   
			for i in range(101):
				#temp = {}
				temp = self.composite_cdf(alpha, curr_dict_conv, curr_dict_comon)
				s1, t1, freq_comp, diff_comp,rmse_1 = self.percentile_difference(curr_dict_route, temp)
				key = (group,round(alpha,2),s1)
				new_dict = {}
				m = sorted(curr_dict_route.keys(), reverse = False)
				for i in range(len(m)):
					perc = m[i]
					val = [temp[perc],curr_dict_route[perc],curr_dict_comon[perc],curr_dict_conv[perc],diff_comp[perc],
							diff_conv[perc],diff_comon[perc],rmse_1,rmse_2,rmse_3,s1,s2,s3]
					new_dict[perc]= val
				
				update_dict[key]= new_dict
				alpha += 0.01
				temp = {}

			all_data[group] = update_dict

		return all_data

	def final_canidate(self , sheet_name):
		print "Finding final canidate..."
		data = self.test_significance() 
		folder_name = sheet_name
		script_dir = os.path.dirname(os.path.abspath(folder_name))
		dest_dir = os.path.join(script_dir, folder_name)
		try:
			os.makedirs(dest_dir)
		except OSError:
			pass # already exists

		for group in data:
			#print group
			curr_regime = data[group]
			m = curr_regime.keys()
			max_score = max(m, key = lambda x:x[2])
			#print 'max_score: ' + str(max_score)
			poss_candidate = []
			other = []
			
			for i in range(len(m)):
				curr_key = m[i]
				if curr_key[2] == max_score[2]:
					poss_candidate.append(curr_key)
				else:
					other.append(curr_key)

			#print "Max Score",max_score
			#print poss_candidate
			final_cand = min(poss_candidate, key = lambda x:x[1])
			#print final_cand
			curr_dict = curr_regime[final_cand]
			val = str(final_cand[0])+ "_"+ str(100*(final_cand[1]))
			#print "val_1: " + str(val)
			results = val
			self.data_write(curr_dict,val, dest_dir,";")

	def data_write(self,dict_,val,dest_dir,del_):
		"""Takes dictionary (key = percentile, val = travel_rate) and user specified
		excel worksheet name as input arguments and outputs those dicitonary values"""
		print "Saving Data as " + val + ".csv ..."
		path = os.path.join(dest_dir, val+".csv")
		with open(path, 'wb') as f:
			writer = csv.writer(f)
			writer = csv.writer(f, delimiter=del_)
			writer.writerow((['Percentile'],['composite_tt','tt_39_11','tt_comon','tt_conv',
                                    'perc_error_comp','perc_error_conv','perc_comon','rmse_1',
                                    'rmse_2','rmse_3','s1','s2','s3']))
			for row in sorted(dict_.items(), key = lambda x:x[0]):
				writer.writerow(row)

	def main(self, sheet_name,type_ = 'time' ):
		self.get_times_data(type_)
		self.get_link_cdfs()
		self.actual_time_cdf()
		self.route_cdf_comonotonic()
		self.route_cdf_convolve()
		#self.final_canidate('Beatty_Backward_Final_')
		#print "Done"
		#return self.test_significance()
		self.final_canidate(sheet_name)

	def save_file(self, filename, dict_, del_):
		with open(filename + ".csv", 'wb') as f:
		    writer = csv.writer(f)
		    writer = csv.writer(f, delimiter=del_)
		    for row in dict_.iteritems():
		        writer.writerow(row)


# streets = {"Centre":['106','107','113',('112','115')], "Highland":[('120','129'),'136','135','134'], 'Penn':['139', '140', ('128', '130'), ('120', '129')],"Baum":['104',('103','109'),'102']}
# links = {'PennOne':['139','140',('128','130')], 'PennTwo':['140',('128','130'),('120', '129')], 'Three':['116',('115','112'),'113']}
# da = DataAnalysis(pathname)
# forward, backward, foward_time, backward_time = da.load_segment([('109','103'), '110', '140'], type0_ = 'time')
# da.save_file('Beatty_Forward', forward, ';')
# da.save_file('Beatty_Backward', backward, ';')



da = Convolve(['140',('128','130'),('120', '129')])
da.main('PennTwo')
