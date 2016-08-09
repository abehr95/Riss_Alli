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


	def composite_cdf(self,alpha, data_type_1,data_type_2, beta = None, data_type_3 = None):
		"""This method takes alpha (weight factor for convolution) as an input parameter
		and computes composite CDF (by method of convolution) of synthesized route CDFs
		obtained by convolution(stratafied monte carlo sampling), and by adding percentiles"""
		#print "COMPOSITE CDF"
		temp = {}
		composite = {}
		n1 = int(round(10000*alpha,0))
		#print n1
		if self.counter:
			n2 = int(round(10000*beta,0))
			n3 = 10000 - n1 - n2
		else:
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
		if self.counter:
			for i in range(n3):
				r3 = round(random.uniform(0,1),2)
				#print r1
				#if r1 == 0:
				#	r1 = 0.01
				tt_3 = data_type_3[r3]
				self.dictionary_update(temp, tt_3)

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
			if curr_key >= -5.0 and curr_key <= 5.0:
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
		all_data = {}
		if self.counter:
			print "Testing Significance..."
			for group in self.actual_cdfs:
				alpha = 0
				curr_dict_comon = self.comon_cdfs[group]
				curr_dict_conv = self.convo_cdfs[group]
				curr_dict_count = self.count_cdfs[group]
				curr_dict_route = self.actual_cdfs[group]
				update_dict = {}
				print group
				s2, t2, freq_conv, diff_conv, rmse_2 = self.percentile_difference(curr_dict_route, curr_dict_conv)
					#print 's2: ' + str(s2)
				s3, t3, freq_comon, diff_comon, rmse_3 = self.percentile_difference(curr_dict_route, curr_dict_comon)   
				s4, t4, freq_count, diff_count, rmse_4 = self.percentile_difference(curr_dict_route, curr_dict_count) 
				for i in range(101):
					brange = 101-i
					beta = 0
					for j in range(brange):
					#temp = {}
						temp = self.composite_cdf(alpha, curr_dict_conv, curr_dict_comon, beta = beta, data_type_3 = curr_dict_count)
						s1, t1, freq_comp, diff_comp, rmse_1 = self.percentile_difference(curr_dict_route, temp)
						key = (group,round(alpha,2),round(beta,2),s1)
						new_dict = {}
						m = sorted(curr_dict_route.keys(), reverse = False)
						for k in range(len(m)):
							perc = m[k]
							val = [temp[perc],curr_dict_route[perc], curr_dict_comon[perc], curr_dict_conv[perc], curr_dict_count[perc], diff_comp[perc],
									diff_conv[perc],diff_comon[perc], diff_count[perc], rmse_1,rmse_2,rmse_3, rmse_4, s1,s2,s3, s4]
							new_dict[perc]= val
						
						update_dict[key]= new_dict
						beta += 0.01
						temp = {}
					alpha += 0.01
				all_data[group] = update_dict

		else:
			print "Testing Significance..."
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
					s1, t1, freq_comp, diff_comp, rmse_1 = self.percentile_difference(curr_dict_route, temp)
					key = (group,round(alpha,2),s1)
					new_dict = {}
					m = sorted(curr_dict_route.keys(), reverse = False)
					for k in range(len(m)):
						perc = m[k]
						val = [temp[perc],curr_dict_route[perc], curr_dict_comon[perc], curr_dict_conv[perc], diff_comp[perc],
								diff_conv[perc],diff_comon[perc], rmse_1,rmse_2,rmse_3, s1,s2,s3]
						new_dict[perc]= val
					
					update_dict[key]= new_dict
					temp = {}
					alpha += 0.01
			all_data[group] = update_dict

		return all_data

	def final_canidate(self , sheet_name):
		if self.counter:
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
				max_score = max(m, key = lambda x:x[3])
				#print 'max_score: ' + str(max_score)
				poss_candidate = []
				other = []
				
				for i in range(len(m)):
					curr_key = m[i]
					if curr_key[3] == max_score[3]:
						poss_candidate.append(curr_key)
					else:
						other.append(curr_key)

				#print "Max Score",max_score
				#print poss_candidate
				final_alpha = min(poss_candidate, key = lambda x:x[1])
				poss_betas = []

				for i in range(len(poss_candidate)):
					curr_key = poss_candidate[i]
					if curr_key[3] == max_score[3]:
						poss_betas.append(curr_key)

				final_cand = min(poss_betas, key = lambda x:x[2])

				#print final_cand
				curr_dict = curr_regime[final_cand]
				val = str(final_cand[0])+ "_"+ str(100*(final_cand[1])) + "_" + str(100*(final_cand[2])) + "_"
				#print "val_1: " + str(val)
				results = val
				self.data_write(curr_dict,val,dest_dir)
				self.get_composite_data(group,curr_dict)
				self.plot_cdfs(group,str(100*(final_cand[1])) + "_" + str(100*(final_cand[2])),dest_dir)

			path = os.path.join(dest_dir, folder_name + "_Plot.jpg")
			self.cluster_object.save_plot(path)

		if self.counter == False:
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
				val = str(final_cand[0])+ "_"+ str(100*(final_cand[1])) + "_" 
				#print "val_1: " + str(val)
				results = val
				self.data_write(curr_dict,val,dest_dir)
				self.get_composite_data(group,curr_dict)
				self.plot_cdfs(group,str(100*(final_cand[1])) + "_",dest_dir)

			path = os.path.join(dest_dir, folder_name + "_Plot.jpg")
			self.cluster_object.save_plot(path)

	def get_composite_data(self,group,test_dic):
		cdf = {}
		for perc in sorted(test_dic.keys()):
			comp = test_dic[perc][0]
			cdf[perc] = comp
		self.comp_cdfs[group] = cdf

	def data_write(self,dict_,val,dest_dir):
		if self.counter:
			print "Saving Data as " + val + str(dict_.values()[0][13]) + ".xls ..."
			wb = Workbook()
			links_name  = ''
			for link in self.segment:
				if type(link) == tuple:
					link = link[0]
				links_name = links_name + link + '_'
			results = wb.add_sheet(links_name)
			self.write_along_column(results, ['Percentile','composite_tt','tt_39_11','tt_comon','tt_conv', 'tt_count',
				'perc_error_comp','perc_error_conv','perc_comon','perc_count','rmse_1',
				'rmse_2','rmse_3', 'rmse_4', 's1','s2','s3', 's4'], 0)

			row = 1
			for key in sorted(dict_.keys()):
				data = [key] + dict_[key]
				self.write_along_column(results, data, row)
				row += 1

			path = os.path.join(dest_dir, 'Cluster'+ val + str(dict_.values()[0][13])+'.xls')
			wb.save(path)

		if self.counter == False:
			print "Saving Data as " + val + str(dict_.values()[0][10]) + ".xls ..."
			wb = Workbook()
			links_name  = ''
			for link in self.segment:
				if type(link) == tuple:
					link = link[0]
				links_name = links_name + link + '_'
			results = wb.add_sheet(links_name)
			self.write_along_column(results, ['Percentile','composite_tt','tt_39_11','tt_comon','tt_conv',
				'perc_error_comp','perc_error_conv','perc_comon', 'rmse_1',
				'rmse_2','rmse_3', 's1','s2','s3'], 0)

			row = 1
			for key in sorted(dict_.keys()):
				data = [key] + dict_[key]
				self.write_along_column(results, data, row)
				row += 1

			path = os.path.join(dest_dir, 'Cluster'+ val + str(dict_.values()[0][10])+'.xls')
			wb.save(path)

	def main(self, sheet_name,type_ = 'time', counter = True ):
		self.counter = counter
		#self.get_times_data(type_)
		self.get_cluster_data()
		self.get_link_cdfs()
		self.actual_time_cdf()
		self.route_cdf_comonotonic()
		self.route_cdf_convolve()
		if self.counter:
			self.route_cdf_countermonotonic()
		#self.final_canidate('Beatty_Backward_Final_')
		#print "Done"
		#return self.test_significance()
		self.final_canidate(sheet_name)

	def plot_cdfs(self,group,val,dest_dir):
		perc = sorted(self.actual_cdfs[group].keys())
		actual = self.get_perc_list(perc, self.actual_cdfs[group])
		comp = self.get_perc_list(perc, self.comp_cdfs[group])
		como = self.get_perc_list(perc,self.comon_cdfs[group])
		convo = self.get_perc_list(perc, self.convo_cdfs[group])
		if self.counter:
			count = self.get_perc_list(perc, self.count_cdfs[group])
		plt.plot(actual, perc, 'b', label = "Actual CDF")
		plt.plot(comp, perc, 'g', label = "Composite CDF")
		plt.plot(como, perc, 'm', label = "Comonotonic CDF")
		plt.plot(convo,perc, 'r', label = "Convolution CDF")
		if self.counter:
			plt.plot(count,perc, 'y', label = "Countermonotonic CDF")
		plt.legend(loc = 4)
		path = os.path.join(dest_dir, str(group) +"_"+ val + "_CDF_plot.jpg")
		plt.xlabel("Time(minutes)")
		plt.ylabel("Percentile")
		plt.savefig(path)
		plt.show()

	def get_perc_list(self, perc, dict_):
		cdf_list = []
		for per in perc:
			cdf_list.append(dict_[per])
		return cdf_list


	def write_along_column(self,sheet, vals, r, c = 0):
		#print "sheet", sheet
		#print "vals" , vals
		#print "r", r
		for i in xrange(len(vals)):
			sheet.write(r, c+i, vals[i])

da = Convolve(['139', '140', ('128', '130'), ('120', '129')])
da.main('Penn_Negative')


