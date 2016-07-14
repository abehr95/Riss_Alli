from decimal import*
#from matplotlib.dates import date2num
from datetime import date, datetime, time
from xlrd import open_workbook, xldate_as_tuple
from xlwt import Workbook, XFStyle
import math
import bisect
import random
from operator import itemgetter, attrgetter



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
		self.keys_sorted = sorted(self.dic.keys())
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
		for time in self.dic[self.keys_sorted[0]]:
			self.links[links] = []
			links += 1
		for key in self.keys_sorted:
			k,v = key, self.dic[key]
			for i in range(len(self.links)):
				self.links[i+1].append(v[i])

	def write_along_column(self,sheet, vals, r, c = 0):
		for i in xrange(len(vals)):
			sheet.write(r, c+i, vals[i])

	# def data_read(self):
	# 	book = open_workbook("I-5 S BlueStats 11-10-9-39.xls")
	# 	sheet = book.sheet_by_name('Data-1')
	# 	raw_data = {}
	# 	trav_times = {}
	# 	outlier_flag = {}
	# 	rec_no = 0
	# 	for row in xrange(1, sheet.nrows):
	# 		mac_id = sheet.cell(row, 0).value
	# 		d_n1 = sheet.cell(row, 2).value
	# 		t_n1 = sheet.cell(row, 3).value
	# 		d_n2 = sheet.cell(row, 5).value
	# 		t_n2 = sheet.cell(row, 6).value
	# 		tt_lnk1 = sheet.cell(row, 7).value
	# 		d_n3 = sheet.cell(row, 9).value
	# 		t_n3 = sheet.cell(row, 10).value
	# 		tt_lnk2 = sheet.cell(row, 11).value
	# 		d_n4 = sheet.cell(row, 13).value
	# 		t_n4 = sheet.cell(row, 14).value
	# 		tt_lnk3 = sheet.cell(row, 15).value
	# 		rec_no += 1
	# 		val = [mac_id, d_n1, t_n1, d_n2, t_n2, tt_lnk1, d_n3, t_n3,
	# 				tt_lnk2, d_n4, t_n4, tt_lnk3]
	# 		raw_data[rec_no] = val
	# 		trav_times[rec_no] = [rec_no, tt_lnk1, tt_lnk2, tt_lnk3]
	# 		self.dic = trav_times



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

	# def find_avg(self, i):
	# 	tot = 0.
	# 	for trip in self.dic:
	# 		tot += self.dic[trip][i]
	# 	return float(tot/len(self.dic))


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
	def __init__(self, data, del_lb, del_ub, t_lb, t_ub, t_min, min_times, trav_times):
		self.data = data
		self.del_lb = del_lb
		self.del_ub = del_ub
		self.t_lb = t_lb
		self.t_ub = t_ub
		self.t_min = t_min
		self.min_times = min_times
		self.trav_times = trav_times
		#self.keys = keys
		self.flag_res = []
		self.data_new = []
		self.outlier_flagged = {}

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
		for i in range(len(self.data)):
			tt = self.data[i]
			for j in range(len(tt)):
				if self.flag_res[j] == 0:
					#print "upper", self.del_ub[j]
					if tt[j] > 0.9*self.del_ub[j]:
						self.del_ub[j]= 1.2*self.del_ub[j]
						#print "Upper raised", self.del_ub[j]
					else:
						self.del_ub[j]= 0.9*self.del_ub[j]
						#print "upper lowered", self.del_lb[j]

					if tt[j] < 1.1*self.t_lb[j]:
						self.del_lb[j] = 1.1*self.del_lb[j]
					else:
						self.del_lb[j] = 0.9*self.del_lb[j]
		#	print "del_ub", self.del_ub
                    
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
				#	print "outlier", tt[j], "mt", self.min_times[j], "lb",self.t_lb[j], "ub",self.t_ub[j]
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
			rec_no = self.find_key(self.trav_times, tt)
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

	def get_outliers(self):
		return self.outlier_flagged

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
		self.add_results(self.outlier_flagged)
		self.new_data()

	def find_key(self, dic, val):
		# return the key of dictionary dic given the value
		return [k for k, v in dic.iteritems() if v == val][0]       

class Final:

	def __init__(self, dict_):
		self.tv_times = dict_
		self._absmin = None
		self.new_trips = {}
		self.removed = {} 
		self.outliers = {}

	def filter_times(self):
		m = InitialParams(self.tv_times)
		m.initialize()
		#print 'len_trav_data: ' + str(len(trav_times))
		self._absmin = m.get_abs_min_times()
		#print "abs min", self._absmin
		_mintimes = m.get_min_times()
		#print "Mintimes", self._mintimes
		_Lb = m.get_lower_bound()
		#print "lower_bound", self._Lb
		_Ub = m.get_upper_bound()
		#print "upper_bound", self._Ub
		_Lt = m.get_accepted_low_times()
		#print "Low time", self._Lt
		_Ut = m.get_accepted_high_times()
		#print "high times", self._Ut
		temp = []
		keys = []
		key_sorted = sorted(self.tv_times.keys())
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
					#print i, len(key_sorted)
					key = key_sorted[i]
					keys.append(key)
					val = self.tv_times[key]
					temp.append(val)

				# if start_rec > (len(key_sorted)-5):
				# 	if last:
				# 		fi = temp[-1]
				# 		for i in range(10-len(temp)):
				# 			temp.append(fi)
				# 		last = False

				#print start_rec, last_rec
		#		print "temp", temp
				if len(temp)> 1:
					count += 1
				#	print temp
					data = Filter(temp, _Lb, _Ub, _Lt, _Ut, _mintimes, self._absmin, self.tv_times)
					data.run_filter()
					temp = data.get_new_data()
					for key in data.get_outliers():
						self.outliers[key] = data.get_outliers()[key]
					#print 'temp: ' + str(len(temp))
					_Lb = data.get_curr_low_bound()
					_Ub = data.get_curr_high_bound()
					_Lt = data.get_curr_accepted_low_times()
					_Ut = data.get_curr_accepted_up_times()
					_mintimes = data.get_curr_min_times()

					if count == 1:
						start_rec += 10
						last_rec += 5
					else:
						start_rec += 5
						last_rec += 5

				if last_rec >= len(key_sorted):
					last_rec = start_rec + (len(key_sorted)- start_rec)


	def remove_outliers(self):
		for rec in self.outliers:
			flags = self.outliers[rec]
			data = self.tv_times[rec]
			#print data
			#print flags
			if all(x==0 for x in flags):
					self.new_trips[rec] = data
			else:
				self.removed[rec] = data

	def get_new_trips(self):
		self.filter_times()
		self.remove_outliers()
		return self.new_trips, self.removed


		#print 'len_outlier_dic: ' + str(len(outlier_flagged))
		#data_write()
