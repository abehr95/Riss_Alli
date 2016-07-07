from decimal import*
#from matplotlib.dates import date2num
from datetime import date, datetime, time
from xlrd import open_workbook, xldate_as_tuple
from xlwt import Workbook, XFStyle
import math
import bisect
import random
from operator import itemgetter, attrgetter

trav_times = {}
outlier_flagged = {}

class InitialParams(object):
	"""
	Initializes parameters 
	Attr:
		dic: dictionary containing travel times
		key = rec_no
		val = [t1, t2, t3]
		self.link1 = list containing travel times of link 1
		self.link2 = list containing travel times of link 2
		self.link3 = list containing travel times of link 3
		self.abs_min = list containing abs_min_times
		self.del_lb = initial lower bound values
		self.del_ub = initial upper bound values
		self.t_lb = intial accepted lower bound on trav_times
		self.t_ub = initial accepted upper bound on trav_times
		self.t_min = initial list of min_times
	"""
	def __init__(self, dic):
		self.dic = dic
		self.links = {}
		self.abs_min = []
		self.del_lb = []
		self.del_ub = []
		self.t_lb = []
		self.t_ub = []
		self.t_min = []

	def list_times(self):
		# adds travel times to respective link lists
		links = 1
		for time in self.dic[self.dic.keys()[0]]:
			self.links[links] = []
			links += 1
		for(k,v)in self.dic.items():
			for i in range(len(self.links)):
				self.links[i+1].append(v[i])

	def compute_minimum(self):
		# computes absolute minimum travel times
		for i in range(len(self.links)):
			l_min = self.percentile(sorted(self.links[i+1]), .001)
			self.abs_min.append(l_min)

	def set_min_times(self):
		# set min times
		for i in range(len(self.links)):
			max_ = max(self.links[i+1])
			self.t_min.append(max_)

	def set_bounds(self):
		# set lower and upper bounds
		for i in range(len(self.links)):
			self.del_lb.append(5)
			self.del_ub.append(10)

	def set_travtime_bounds(self):
		# set bounds on accepted travel times
		for i in range(len(self.links)):
			tt_lb = self.t_min[i] - self.del_lb[i]
			tt_ub = self.t_min[i] + self.del_ub[i]
			self.t_lb.append(tt_lb)
			self.t_ub.append(tt_ub)

	def get_abs_min_times(self):
		# returns results containing absolute min travel times
		return self.abs_min

	def get_min_times(self):
		# returns initial min_times
		return self.t_min

	def get_lower_bound(self):
		# returns lower bound
		return self.del_lb

	def get_upper_bound(self):
		# returns upper bound
		return self.del_ub   

	def get_accepted_low_times(self):
		# returns lower bound on accepted travel times
		return self.t_lb

	def get_accepted_high_times(self):
		# returns upper bound on accepted travel times
		return self.t_ub

	def initialize(self):
		# main() method for this class
		self.list_times()
		self.compute_minimum()
		self.set_bounds()
		self.set_min_times()
		self.set_travtime_bounds()        
    
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
		f = math.floor(k)
		c = math.ceil(k)
		if f == c:
			return (N[int(k)])
		d0 = (N[int(f)]) * (c-k)
		d1 = (N[int(c)]) * (k-f)
		return d0+d1

class Filter:
	"""
	This class flags outliers
	Attr:
		data = list of lists containing travel times for three links.
			the length of this dataset is 10
			each rec in data is of type [t1, t2, t3]
		del_lb = row vector containing lower bound
		del_ub = row vector containing uppper bound
		t_lb = row vector containing lower bound of accepted travel times
		t_ub = row vector containing upper bound of accepted travel times
		t_min = row vector containing current minimum travel times
		min_times = row vector containing absolute minimum travel times        
	"""
	def __init__(self, data, del_lb, del_ub, t_lb, t_ub, t_min, min_times):
		self.data = data
		self.del_lb = del_lb
		self.del_ub = del_ub
		self.t_lb = t_lb
		self.t_ub = t_ub
		self.t_min = t_min
		self.min_times = min_times
		#self.keys = keys
		self.flag_res = []
		self.data_new = []

	def set_tmin(self):
		"""
		This method re-evaluates minimum travel vector
		Goes through records j = 4-7 and checks for the following condition
		if tt(j) > absolute min_time and tt(j) < current min travel time
		then set t_min = tt(j)
		"""
		#pass
		if len(self.data) == 10:
			ld = len(self.data[0])
			check = self.data[3:7]
			new_min = [1000 for i in range(len(self.data[0]))]
			for tt in check:
				#print 'tt: ' + str(tt)
				for j in range(len(tt)):
					if tt[j] >= self.min_times[j] and tt[j] < new_min[j]:
						new_min[j] = tt[j]

			for i in range(ld):
				if new_min[i] != 1000:
					self.t_min[i] = new_min[i]
                    

	def update_boundaries(self):
		"""
		This method updates del_time boundaries
		for every travel time record in data and checks two separate
		conditions to update lower and upper boundaries.
		For updating upper bound:
		if tt(j) > 0.9*curr_upper_bound:
			set curr_upper_bound to 1.2*curr_upper_bound
		else:
			set curr_upper_bound to 0.9*curr_upper_bound
		
		For updating lower bound:
		if tt(j) < 1.1*curr_lower_bound:
			set curr_lower_bound to 1.1*curr_lower_bound
		else:
			set curr_lower_bound to 0.9*curr_lower_bound            
		"""
        #pass
		for tt in self.data:
			for j in range(len(tt)):
				if self.flag_res[j] == 0:
					if tt[j] > 0.9*self.t_ub[j]:
						self.del_ub[j]= 1.2*self.del_ub[j]
					else:
						self.del_ub[j]= 0.9*self.del_ub[j]

					if tt[j] < 1.1*self.t_lb[j]:
						self.del_lb[j] = 1.1*self.del_lb[j]
					else:
						self.del_lb[j] = 0.9*self.del_lb[j]
                    
	def update_travtime_bounds(self):
		"""
		This method updates accepted bounds of travel times
		tt_lb = max(0, tt_min-del_lb)
		tt_ub = max(0, tt_min_del_ub)
		"""
		#pass
		ld = len(self.data[0])
		for i in range(ld):
			tt_min = self.t_min[i]
			del_lb = self.del_lb[i]
			del_ub = self.del_ub[i]
			tt_lb = max(0, tt_min-del_lb)
			tt_ub = max(0, tt_min+del_ub)
			self.t_lb[i] = tt_lb
			self.t_ub[i] = tt_ub

	def outlier_flag(self):
		"""
		This method checks for outliers
		For every record in data
		if tt(j) < current accepted lower bound or > curr_accepted_ub:
			flag this record as an outlier (flag = 1)
		else:
			set flag = 0 (not an outlier)
		append outlier flag to flag_res
		"""
		#pass
		for tt in self.data:
			flag = []
			for j in range(len(tt)):
				if tt[j] < self.min_times[j] or tt[j] < self.t_lb[j] or tt[j] > self.t_ub[j]:
					flag.append(1)
				else:
					flag.append(0)
			self.flag_res.append(flag)

	def add_results(self, dic):
		"""
		Adds outlier flags to a dictionary
		Attr:
			dic = dictionary into which these results should be written
			rec_no = this is a common key for all dictionaries
			dic[rec_no] = flag
		"""
		#pass
		for i in range(len(self.data)):
			tt = self.data[i]
			flag = self.flag_res[i]
			rec_no = self.find_key(trav_times, tt)
			#rec_no = self.keys[i]
			#print 'rec_no: ' + str(rec_no)
			#print 'len_flag: ' + str(len(self.flag_res))
			dic[rec_no] = flag

	def new_data(self):
		"""
		This method appends records 6-10 into list data_new
		this list is used as overlapping 5 in next iteration
		"""
		if len(self.data) == 10:
			for i in range(5):
				tt = self.data[5+i]
				self.data_new.append(tt)

	def get_new_data(self):
		# this method returns 5 elements that will be part of overlapping 5
		return self.data_new

	def get_curr_low_bound(self):
		# this method returns current bounds
		return self.del_lb

	def get_curr_high_bound(self):
		# returns current high bound
		return self.del_ub

	def get_curr_accepted_low_times(self):
		# this method returns current accepted lower bounds on travel times
		return self.t_lb

	def get_curr_accepted_up_times(self):
		# this method returns curent accepted upper bounds on travel times
		return self.t_ub

	def get_curr_min_times(self):
		# this method returns current minimum travel times
		return self.t_min

	def run_filter(self):
		# this is the main() method for running the filtering algorithm
		self.set_tmin()
		self.update_travtime_bounds()
		self.outlier_flag()
		self.update_boundaries()
		self.add_results(outlier_flagged)
		self.new_data()

	def find_key(self, dic, val):
		# return the key of dictionary dic given the value
		return [k for k, v in dic.iteritems() if v == val][0]       

class Final:

	def __init__(self, dict_):
		self.tv_times = dict_
		self._absmin = None
		self._mintimes = None
		self._Lb = None
		self._Ub = None
		self._Lt = None
		self._Ut = None
		self.data = None
		self.new_trips = {}
		self.removed = {}

	def fill_travel_time(self):
		for key in self.tv_times:
			trav_times[key] = self.tv_times[key]

	def filter_times(self):
		m = InitialParams(trav_times)
		m.initialize()
		print 'len_trav_data: ' + str(len(trav_times))
		self._absmin = m.get_abs_min_times()
		self._mintimes = m.get_min_times()
		self._Lb = m.get_lower_bound()
		self._Ub = m.get_upper_bound()
		self._Lt = m.get_accepted_low_times()
		self._Ut = m.get_accepted_high_times()
		temp = []
		keys = []
		key_sorted = sorted(trav_times.keys())
		start_rec = 0
		last_rec = 10
		count = 0
		#last = True

		if start_rec > len(key_sorted):
			return
		else:
			for j in xrange(len(key_sorted)):
				#print start_rec, last_rec, len(key_sorted)
				for i in xrange(start_rec, last_rec):
					key = key_sorted[i]
					keys.append(key)
					val = trav_times[key]
					temp.append(val)

				# if start_rec > (len(key_sorted)-5):
				# 	if last:
				# 		fi = temp[-1]
				# 		for i in range(10-len(temp)):
				# 			temp.append(fi)
				# 		last = False

				#print start_rec, last_rec
				if len(temp)> 1:
					count += 1
					#print temp
					self.data = Filter(temp, self._Lb, self._Ub, self._Lt, self._Ut, self._mintimes, self._absmin)
					self.data.run_filter()
					temp = self.data.get_new_data()
					#print 'temp: ' + str(len(temp))
					self._Lb = self.data.get_curr_low_bound()
					self._Ub = self.data.get_curr_high_bound()
					self._Lt = self.data.get_curr_accepted_low_times()
					self._Ut = self.data.get_curr_accepted_up_times()
					self._mintimes = self.data.get_curr_min_times()

					if count == 1:
						start_rec += 10
						last_rec += 5
					else:
						start_rec += 5
						last_rec += 5

				if last_rec >= len(key_sorted):
					last_rec = start_rec + (len(key_sorted)- start_rec-1)


	def remove_outliers(self):
		for rec in outlier_flagged:
			flags = outlier_flagged[rec]
			data = trav_times[rec]
			#print data
			#print flags
			if all(x==0 for x in flags):
					self.new_trips[rec] = data
			else:
				self.removed[rec] = data

	def get_new_trips(self):
		self.fill_travel_time()
		self.filter_times()
		self.remove_outliers()
		return self.new_trips, self.removed


		#print 'len_outlier_dic: ' + str(len(outlier_flagged))
		#data_write()
