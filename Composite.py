import numpy as np
import os
import math
import random
import matplotlib.pyplot as plt
from xlwt import Workbook, XFStyle


class Composite:

	def __init__(self, folder, segment, act, comon, convo, count = None):
		self.actual = act
		self.comonotonic = comon
		self.convolution = convo
		self.folder_name = folder
		self.segment = segment
		self.composite_pos = {}
		self.composite_neg = {}
		if count:
			self.countermonotonic = count

	def dictionary_update(self,dict_type, key):
		"""Takes user specified name of dictionary and key as input argument and
		generates dictionary (key = Individual vehicle travel rate generataed
		from each Monte Carlo run, value = frequency of that travel rate) for various
		operating conditions"""
		
		if key in dict_type:
			dict_type[key] += 1
		else:
			dict_type[key]= 1

	def percentile_difference(self, dict_act, dict_other):
		"""Takes CDF tables as input arguments and computes percentage error between
		individual percentile values"""
		freq_table = {}
		diff_table = {}
		m = sorted(dict_act.keys(), reverse = False)
		square_error = 0
		for i in range(len(m)):
			key = m[i]
			perc_diff = 100*((dict_other[key]- dict_act[key])/dict_act[key])
			perc_error = round(perc_diff,2)
			square_error += round((dict_other[key]-dict_act[key])**2, 3)
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

	def count_record(self,dict_type):
		L = sorted(dict_type.viewkeys(), reverse = False)
		n = 0
		for i in xrange(len(L)):
			z = L[i]
			temp_1 = dict_type[z]
			n += temp_1
		return n

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

	def composite_cdf_positive(self,alpha,data_type_1,data_type_2):
		"""This method takes alpha (weight factor for convolution) as an input parameter
		and computes composite CDF (by method of convolution) of synthesized route CDFs
		obtained by convolution(stratafied monte carlo sampling), and by adding percentiles"""
		#print "COMPOSITE CDF"
		temp = {}
		composite = {}
		n1 = int(round(100000*alpha,0))
		#print n1
		n2 = 100000 - n1
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

	def composite_cdf_negative(self,alpha, data_type_1,data_type_2, beta, data_type_3):
		"""This method takes alpha (weight factor for convolution) as an input parameter
		and computes composite CDF (by method of convolution) of synthesized route CDFs
		obtained by convolution(stratafied monte carlo sampling), and by adding percentiles"""
		#print "COMPOSITE CDF"
		temp = {}
		composite = {}
		n1 = int(round(30000*alpha,0))
		n2 = int(round(30000*beta,0))
		n3 = 30000 - n1 - n2

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

	def test_significance_neg(self):
		"""This method convolves synthesized CDFs sampling each CDF on a weighted
		percentage alpha(alpha varies between 0-1), and performs KS Test to check for
		statistical significance between actual_cdf and newly synthesized cdf"""
		all_data = {}
		print "Testing Significance..."
		for group in self.actual:
			alpha = 0
			curr_dict_comon = self.comonotonic[group]
			curr_dict_conv = self.convolution[group]
			curr_dict_count = self.countermonotonic[group]
			curr_dict_route = self.actual[group]
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
					temp = self.composite_cdf_negative(alpha, curr_dict_conv, curr_dict_comon, beta, curr_dict_count)
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

		return all_data

	def final_canidate_neg(self):
		print "Finding final canidate..."
		data = self.test_significance_neg() 
		script_dir = os.path.dirname(os.path.abspath(self.folder_name))
		dest_dir = os.path.join(script_dir, self.folder_name)

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
			self.data_write_neg(curr_dict,val,dest_dir)
			self.get_composite_data(group,curr_dict, self.composite_neg)
			self.plot_cdfs_neg(group,str(100*(final_cand[1])) + "_" + str(100*(final_cand[2])) + "_Negative",dest_dir)

	def test_significance_pos(self):
		"""This method convolves synthesized CDFs sampling each CDF on a weighted
		percentage alpha(alpha varies between 0-1), and performs KS Test to check for
		statistical significance between actual_cdf and newly synthesized cdf"""
		print "Testing Significance..."
		all_data = {}
		for group in self.actual:
			alpha = 0
			curr_dict_comon = self.comonotonic[group]
			curr_dict_conv = self.convolution[group]
			curr_dict_route = self.actual[group]
			update_dict = {}
			print group
			s2, t2, freq_conv, diff_conv, rmse_2 = self.percentile_difference(curr_dict_route, curr_dict_conv)
				#print 's2: ' + str(s2)
			s3, t3, freq_comon, diff_comon, rmse_3 = self.percentile_difference(curr_dict_route, curr_dict_comon)   
			for i in range(101):
				#temp = {}
				temp = self.composite_cdf_positive(alpha, curr_dict_conv, curr_dict_comon)
				s1, t1, freq_comp, diff_comp, rmse_1 = self.percentile_difference(curr_dict_route, temp)
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

	def final_canidate_pos(self):
		print "Finding final canidate..."
		data = self.test_significance_pos() 
		script_dir = os.path.dirname(os.path.abspath(self.folder_name))
		dest_dir = os.path.join(script_dir, self.folder_name)

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
			self.data_write_pos(curr_dict,val,dest_dir)
			self.get_composite_data(group,curr_dict, self.composite_pos)
			self.plot_cdfs_pos(group,100*(final_cand[1]),dest_dir)

	def get_composite_data(self,group,test_dic, final_dic):
		cdf = {}
		for perc in sorted(test_dic.keys()):
			comp = test_dic[perc][0]
			cdf[perc] = comp
		final_dic[group] = cdf

	def data_write_neg(self,dict_,val,dest_dir):
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

		path = os.path.join(dest_dir, 'Cluster'+ val + str(dict_.values()[0][13])+ '_Negative.xls')
		wb.save(path)


	def data_write_pos(self,dict_,val,dest_dir):
		print "Saving Data as " + val + str(dict_.values()[0][10]) + ".xls ..."
		wb = Workbook()
		links_name  = ''
		for link in self.segment:
			if type(link) == tuple:
				link = link[0]
			links_name = links_name + link + '_'
		results = wb.add_sheet(links_name)
		self.write_along_column(results, ['Percentile','composite_tt','tt_39_11','tt_comon','tt_conv',
			'perc_error_comp','perc_error_conv','perc_comon','rmse_1',
			'rmse_2','rmse_3','s1','s2','s3'], 0)

		row = 1
		for key in sorted(dict_.keys()):
			data = [key] + dict_[key]
			self.write_along_column(results, data, row)
			row += 1

		path = os.path.join(dest_dir, 'Cluster'+ val + str(dict_.values()[0][10])+'_Positive.xls')
		wb.save(path)

	def plot_cdfs_pos(self,group,val,dest_dir):
		perc = sorted(self.actual[group].keys())
		actual = self.get_perc_list(perc, self.actual[group])
		comp = self.get_perc_list(perc, self.composite_pos[group])
		como = self.get_perc_list(perc,self.comonotonic[group])
		convo = self.get_perc_list(perc, self.convolution[group])
		plt.show()
		plt.plot(actual, perc, 'b', label = "Actual CDF")
		plt.plot(comp, perc, 'g', label = "Composite CDF")
		plt.plot(como, perc, 'm', label = "Comonotonic CDF")
		plt.plot(convo,perc, 'r', label = "Convolution CDF")
		plt.legend(loc = 4)
		path = os.path.join(dest_dir, str(group) +"_"+ str(val) + "_CDF_plot.jpg")
		plt.xlabel("Time(minutes)")
		plt.ylabel("Percentile")
		plt.title("Cluster "+ str(group) + " CDFs without Countermonotonic")
		plt.savefig(path)
		plt.show()

	def plot_cdfs_neg(self,group,val,dest_dir):
		perc = sorted(self.actual[group].keys())
		actual = self.get_perc_list(perc, self.actual[group])
		comp = self.get_perc_list(perc, self.composite_neg[group])
		como = self.get_perc_list(perc,self.comonotonic[group])
		convo = self.get_perc_list(perc, self.convolution[group])
		count = self.get_perc_list(perc, self.countermonotonic[group])
		plt.plot(actual, perc, 'b', label = "Actual CDF")
		plt.plot(comp, perc, 'g', label = "Composite CDF")
		plt.plot(como, perc, 'm', label = "Comonotonic CDF")
		plt.plot(convo,perc, 'r', label = "Convolution CDF")
		plt.plot(count,perc, 'y', label = "Countermonotonic CDF")
		plt.legend(loc = 4)
		path = os.path.join(dest_dir, str(group) +"_"+ str(val) + "_CDF_plot.jpg")
		plt.xlabel("Time(minutes)")
		plt.ylabel("Percentile")
		plt.title("Cluster "+ str(group) + " CDFs with Countermonotonic")
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

	def save(self):
		self.final_canidate_pos()
		if self.countermonotonic:
			self.final_canidate_neg()