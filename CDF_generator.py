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
from Analysis import DataAnalysis
import matplotlib.pyplot as plt
from Composite import Composite

pathname = os.getcwd()

class Convolve:

	def __init__(self, segment_):
		self.analysis = DataAnalysis(pathname)
		self.cluster_object = None
		self.segment = segment_
		self.times = {}
		self.links = []
		self.longtimes = {}
		self.data = {}
		self.actual_cdfs = {}
		self.link_cdfs = {}
		self.comon_cdfs = {}
		self.convo_cdfs = {}
		self.count_cdfs = {}
		self.comp_cdfs = {}
		self.counter = None
		self.folder_name = None


	def get_times_data(self, type_, sort_ = True):
		print "Accquiring data..."
		self.data, times, self.longtimes = self.analysis.get_long(self.segment, type_ = type_)
		self.cluster_object = self.analysis.c_object
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
				#print "Trips Removed from " + group +":", removed
			self.times = temp

	def get_cluster_data(self, type_ = 'time'):
		self.data, self.times, self.longtimes = self.analysis.cluster_data(self.segment)
		self.cluster_object = self.analysis.c_object

		script_dir = os.path.dirname(os.path.abspath(self.folder_name))
		dest_dir = os.path.join(script_dir, self.folder_name)
		try:
			os.makedirs(dest_dir)
		except OSError:
			pass # already exists

		path = os.path.join(dest_dir, self.folder_name + "_Plot.jpg")
		self.cluster_object.save_plot(path)
		#print "link times", self.times
		#print "total times",self.longtimes
		self.links = sorted(self.times[self.times.keys()[0]].keys())
		#print "c_object", c_object

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

	def route_cdf_countermonotonic(self):
		for group in self.times:
			Ncdf = []
			links = sorted(self.times[group].keys())
			ntrips = len(self.longtimes[group])
			for j in xrange(ntrips):
				tNeg = 0.
				for link in links:
					times = sorted(self.times[group][link])
					if link%2 == 0:
						tNeg = tNeg + times[j]
					else:
						tNeg = tNeg + times[ntrips - j - 1]
				Ncdf.append(tNeg)
				cdf = self.compute_cdf(Ncdf)
			self.count_cdfs[group] = cdf

	def final(self):
		Curr_count = None
		if self.counter:
			Curr_count = self.count_cdfs
		Curr_como = self.comon_cdfs
		Curr_convo = self.convo_cdfs
		Curr_act = self.actual_cdfs
		comp = Composite(self.folder_name, self.segment, Curr_act, Curr_como, Curr_convo, Curr_count)
		comp.save()

	def main(self, sheet_name,type_ = 'time', counter = True ):
		self.folder_name = sheet_name
		self.counter = counter
		#self.get_times_data(type_)
		self.get_cluster_data()
		self.get_link_cdfs()
		self.actual_time_cdf()
		self.route_cdf_comonotonic()
		self.route_cdf_convolve()
		if self.counter:
			self.route_cdf_countermonotonic()
		self.final()

da = Convolve(['139', '140', ('128', '130'), ('120', '129')])
da.main('Penn_Both')


