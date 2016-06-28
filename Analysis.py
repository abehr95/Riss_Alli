from datetime import datetime
from Data_Process import RawDataManager
import numpy as np
import csv
import os
import math
import random

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
		self.organized_R = {}

	def load_data(self,file_):
		self.data_object = RawDataManager(self.path)
		self.data_object.load_sorted_trips(file_)
		self.data = self.data_object.get_valid_trips()

	def get_trips(self,sequence):
		self.forward_trips = self.data_object.check_seq(sequence)
		print "Foward Trips:", len(self.forward_trips)
		reverse = sequence
		reverse.reverse()
		self.backwards_trips = self.data_object.check_seq(reverse)
		print "Backward Trips:", len(self.backwards_trips)

	def find_times(self, timelist):
		if type(timelist[0][1]) == tuple:
			start_time = sorted(timelist[0], key = lambda x: x[-1])[0][0][0]
		else:
			start_time = timelist[0][0][0]
		if type(timelist[-1][1]) == tuple:
			end_time = sorted(timelist[-1], key = lambda x: x[-1])[0][0][0]
		else:
			end_time = timelist[-1][0][0]

		diff = self.diff_time(start_time,end_time)
		return (datetime.utcfromtimestamp(start_time),datetime.utcfromtimestamp(end_time),diff/60.)

	def list_times(self, group, type0_ = None):
		start_times = []
		end_times = []
		totals = []
		for trip in group:
			(start, end, diff) = self.find_times(trip[1])
			if type0_ == 'time':
				start = start.time()
				end = end.time()

			start_times.append(start)
			end_times.append(end)
			totals.append(diff)

		return start_times, end_times, totals

	def diff_time(self, time1, time2):
		start = datetime.utcfromtimestamp(time1)
		end = datetime.utcfromtimestamp(time2)
		total_time = end - start
		return total_time.total_seconds()

	def sort_trips(self,type_ = None):
		if type_ == 'day' or type_ == None:
			self.sort_ft = sorted(self.forward_trips.items(), key = lambda x: x[0][2])
			self.sort_bt = sorted(self.backwards_trips.items(), key = lambda x: x[0][2])

		if type_ == 'time':
			self.sort_ft = sorted(self.forward_trips.items(), key = lambda x: datetime.utcfromtimestamp(x[0][2]).time())
			self.sort_bt = sorted(self.backwards_trips.items(), key = lambda x: datetime.utcfromtimestamp(x[0][2]).time())

	def split_trips(self, group_number, overlap = None):
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
				if overlap < group_number:
					i = i - overlap
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

	def get_avg_stdv(self, timelist):
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
		if type(info[1]) == tuple:
			time = sorted(info, key = lambda x: x[-1])[0][0][0]
		else:
			time = info[0][0]
		return time

	def find_R(self,X, Y):

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
		if type(tuple_[1]) == tuple:
			sensors = []
			for tuples in tuple_:
				sensors.append(tuples[1])
			return sorted(sensors)[0]
		return tuple_[1]

	def split_sensors(self, list_):
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

	def compile_data(self,dict_, type_ = None):
		data = {}
		sensor_times = {}
		for group in dict_:
			Rvalues = {}
			i = 0
			sensorlist = self.split_sensors(dict_[group])
			sensors = self.time_between(sensorlist)
			sensor_times[group] = sensors
			while i < len(sensors)-1:
				(link1,times1) = sensors[i]
				(link2,times2) = sensors[i+1]
				link = 'tt_'+link1+'/' + link2
				Rvalues[link] = self.find_R(times1,times2)
				i = i + 1

			start, end, totals = self.list_times(dict_[group], type_)
			print "Start", start
			print "End", end
			print "Totals", totals
			avgtimes = []
			for sensor in sensors:
				avg, stdv = self.get_avg_stdv(sensor[1])
				avgtimes.append((sensor[0],avg,stdv))

			timespan = self.timespan(start)
			average , stdv = self.get_avg_stdv(totals)

			#, 'Start,End,Total(min)':times, 
			data[group] = {'R':Rvalues, 'Timespan':timespan, 'Segment:(Average time(sec), Average stdv)': avgtimes, 'Average_trip(min)':average, 'Stdv(min)':stdv, 'Totals': totals}

		return data, sensor_times

	def load_segment(self, sensors, type0_ = None):
		self.load_data('SortedTrips.csv')
		self.get_trips(sensors)
		self.sort_trips(type_ = type0_)
		self.split_trips(50, overlap = 25)
		foward, forward_time = self.compile_data(self.forward_split,type_ = type0_)
		backward, backward_time = self.compile_data(self.backward_split,type_ = type0_)
		return foward, backward, forward_time, backward_time

	def get_long(self, sensors, type_ = None, reverse = False):
		forward, backward, forward_time, backward_time = self.load_segment(sensors, type0_ = type_)
		if len(forward) >= len(backward):
			long_ = forward
			longtime = forward_time
			short = backward
			shorttime = backward_time
		else:
			long_ = backward
			longtime = backward_time
			short = forward
			shorttime = forward_time

		if reverse:
			long_ = short
			longtime = shorttime
		return long_, longtime

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

	def organize_R(self, bounds, compiled_data):
		s_bounds = sorted(bounds)
		s_bounds.insert(0,-1)
		s_bounds.append(1.000001)
		o_keys = []
		for i in xrange(len(s_bounds)-1):
			if i+1 == len(s_bounds)-1:
				name = str(s_bounds[i]) + "_to_" +str(1)
			else:
				name = str(s_bounds[i]) + "_to_" +str(s_bounds[i+1])
			self.organized_R[name] = {}
			o_keys.append(name)

		for group in compiled_data:
			Rvalue = compiled_data[group]['R']
			if len(Rvalue) != 1:
				raise ValueError, 'more than one rvalue for ' + group
			timespan = self.timespan(compiled_data[group]['Start,End,Total(min)'])
			average = compiled_data[group]['Average_trip(min)']
			stdv = compiled_data[group]['Stdv(min)']
			data = {'Group':group,'Timespan':timespan,'Average':average,'Stdv':stdv}
			for i in xrange(len(o_keys)):
				if Rvalue[Rvalue.keys()[0]] >= s_bounds[i] and Rvalue[Rvalue.keys()[0]] < s_bounds[i+1]:
					self.organized_R[o_keys[i]][Rvalue[Rvalue.keys()[0]]] = data
					break

	def main(self, streets, type_ = None):
		for street in streets:
			da = DataAnalysis(pathname)
			forward, backward, foward_time, backward_time = da.load_segment(streets[street], type0_ = type_)
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
		self.data = {}
		self.actual_cdfs = {}
		self.link_cdfs = {}
		self.comon_cdfs = {}
		self.convo_cdfs = {}

	def get_times_data(self, type_):
		print "Accquiring data..."
		self.data, times= self.analysis.get_long(self.segment, type_ = type_)
		for group in times:
			self.times[group] = {}
			for sensor_dt in times[group]:
				sensor = sensor_dt[0]
				times_ = sensor_dt[1]
				self.times[group][sensor] = times_

	def actual_time_cdf(self):
		print "Finding actual CDFs..."
		cdfs = {}
		for group in self.data:
			times = self.data[group]['Totals']
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
		temp = {}
		composite = {}
		n1 = int(round(10000*alpha,0))
		n2 = 10000 - n1
		for i in range(n1):
			r1 = round(random.uniform(0,1),2)
			if r1 == 0:
				r1 = 0.01
			tt_1 = data_type_1[r1]
			self.dictionary_update(temp, tt_1)
		for j in range(n2):
			r2 = round(random.uniform(0,1),2)
			if r2 == 0:
				r2 = 0.01 
			tt_2 = data_type_2[r2]
			self.dictionary_update(temp, tt_2)

		composite = self.percentile_freq(temp, composite)
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
#what does this section of code do?? why am i getting all zeros?
		n = sorted(freq_table.keys(), reverse = False)
		tot = 0
		for j in range(len(n)):
			curr_key = n[j]
			if curr_key >= -1.0 and curr_key <= 1.0:
				tot += freq_table[curr_key]
		if tot >= 85:
			test = 1
		else:
			test = 0

		rmse = math.sqrt(square_error/100.0)
		return tot, test, freq_table, diff_table, rmse

	def test_significance_new(self):
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
	        
			for i in range(101):
				#temp = {}
				temp = self.composite_cdf(alpha, curr_dict_conv, curr_dict_comon)
				s1, t1, freq_comp, diff_comp,rmse_1 = self.percentile_difference(curr_dict_route, temp)
	 			#print 's1: ' + str(s1)
				s2, t2, freq_conv, diff_conv, rmse_2 = self.percentile_difference(curr_dict_route, curr_dict_conv)
				#print 's2: ' + str(s2)
				s3, t3, freq_comon, diff_comon, rmse_3 = self.percentile_difference(curr_dict_route, curr_dict_comon)
				#print 's3: ' + str(s3)
	            
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

	def main(self, type_ = None):
		self.get_times_data(type_)
		self.get_link_cdfs()
		self.actual_time_cdf()
		self.route_cdf_comonotonic()
		self.route_cdf_convolve()
		return self.test_significance_new()




streets = {"Centre":['106','107','113',('112','115')], "Highland":[('120','129'),'136','135','134'], 'Penn':['139', '140', ('128', '130'), ('120', '129')],"Baum":['104',('103','109'),'102']}
links = {'PennOne':['139','140',('128','130')], 'PennTwo':['140',('128','130'),('120', '129')], 'Three':['116',('115','112'),'113']}
da = DataAnalysis(pathname)
da.main(streets, type_ = 'time')
