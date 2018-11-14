""" Receiver Operating Characteristic Curve
"""
import matplotlib.pyplot as plt
#from CI_binomial_proportion import *
from numpy.random import randint
from numpy import mean, percentile

__all__ = ["ROCCurve"]

class ROCCurve():
	""" Receiver Operating Characteristic Curve
	"""
	def __init__(self, y_score, y_true):
		self.ROC_curve = self.compute_ROC(y_score, y_true)
		try:
			self.ROC_curve['AUC_CI'] = self.compute_AUC_CI(y_score, y_true, 
				alpha = 0.05, 
				method = 'bootstrap', 
				number_of_bootstrap_replications = 100)
		except:
			pass

	def compute_ROC(self, y_score, y_true):
		""" computes the ROC Curve

		Parameters:
	    -----------
	    y_score : predictions of the classifier which are real numbers
	            : larger value indicates higher confidence that the label is 1
	    y_true  : the true binary class label [1 is considered positive, rest negtive]

	    Returns:
	    --------
	    ROC_curve['true_positive_rate'] : the Y-axis of the ROC Curve [sensitivity]
	                                    : Pr[ y_predicted = 1 | y_true = 1 ]  
	                                    : Pr[ y_score >= threshold | y_true = 1 ]  
	    ROC_curve['false_positive_rate']: the X-axis of the ROC Curve [1-specificity]
	                                    : Pr[ y_predicted = 1 | y_true = 0]  
	                                    : Pr[ y_score >= threshold 1 | y_true = 0]  
	    ROC_curve['threshold']          : the correponding thresholds

		"""

		assert(len(y_score)==len(y_true)),'y_score and y_true should have the same length'

		thresholds = []
		number_of_true_positives  = []
		number_of_false_positives = []

		number_of_positives = 0.0
		number_of_negatives = 0.0

		thresholds.append(float('Inf'))
		number_of_true_positives.append(0.0)
		number_of_false_positives.append(0.0)

		for score, label in sorted( zip(y_score,y_true), reverse = True ):
			thresholds.append(score)
			if label == 1:
				number_of_positives += 1.0
			    # add one to the true positives			
				number_of_true_positives.append(number_of_true_positives[-1]+1.0)
				number_of_false_positives.append(number_of_false_positives[-1])
			else:
				number_of_negatives += 1.0
				# add one to the false positives			
				number_of_true_positives.append(number_of_true_positives[-1])
				number_of_false_positives.append(number_of_false_positives[-1]+1.0)

		ROC_curve = {}
		ROC_curve['threshold']           = thresholds
		if number_of_positives > 0.0:
			ROC_curve['true_positive_rate']  = [value/number_of_positives for value in number_of_true_positives]
		else:
			ROC_curve['true_positive_rate']  = [0.0 for value in number_of_true_positives]		
		if number_of_negatives > 0.0:
			ROC_curve['false_positive_rate'] = [value/number_of_negatives for value in number_of_false_positives]
		else:
			ROC_curve['false_positive_rate'] = [0.0 for value in number_of_false_positives]
		ROC_curve['number_of_positives'] = number_of_positives
		ROC_curve['number_of_negatives'] = number_of_negatives
		ROC_curve['number_of_instances'] = number_of_positives + number_of_negatives
		ROC_curve['AUC'] = self.compute_AUC(ROC_curve,
			FPR_low = 0.0, FPR_high = 1.0)

		return ROC_curve	

	def compute_AUC(self, ROC_curve = None, 
		FPR_low = 0.0, FPR_high = 1.0):
		""" computes the area under the ROC curve between FPR_low to FPR_high

		Parameters:
		-----------
		FPR_low  : defaults to 0.0 
		FPR_high : defaults to 1.0

		Results:
		--------
		AUC : the area under the ROC curve

		"""		
		if ROC_curve is None:
			ROC_curve = self.ROC_curve

		FPR = ROC_curve['false_positive_rate']
		TPR = ROC_curve['true_positive_rate']

		temp = [abs(f-FPR_low) for f in FPR]
		index_low = temp.index(min(temp))

		temp = [abs(f-FPR_high) for f in FPR]
		temp.reverse()
		index_high = len(temp)-1-temp.index(min(temp))

		AUC = 0.0
		for i in xrange(index_low+1, index_high+1):
			base   = FPR[i]-FPR[i-1]
			height = TPR[i]
			AUC   += (base*height)

		return AUC	

	def set_operating_point(self, 
		operating_point_FPR = None,
		operating_point_TPR = None,
		operating_point_threshold = None):
		""" set the operating point
		can specify either the rater TPR, FPR, or directly the threshold
		finds the threshold for a given operating_point_FPR or operating_point_TPR

		Parameters:
		-----------
		operating_point_FPR : target false positive rate
		OR
		operating_point_TPR : target true positive rate
		OR
		operating_point_threshold : target threshold

		Results:
		--------
		self.ROC_curve['operating_point']
		"""
		FPR = self.ROC_curve['false_positive_rate']
		TPR = self.ROC_curve['true_positive_rate']
		threshold = self.ROC_curve['threshold']
	 
		if operating_point_FPR:
			index = 0
			while FPR[index] < operating_point_FPR:
				index += 1

		if operating_point_TPR:
			index = 0
			while TPR[index] < operating_point_TPR:
				index += 1

		if operating_point_threshold:
			index = 0
			while threshold[index] > operating_point_threshold:
				index += 1		

		self.ROC_curve['operating_point'] = {}
		self.ROC_curve['operating_point']['threshold'] = threshold[index]

		self.ROC_curve['operating_point']['false_positive_rate'] = FPR[index]
		self.ROC_curve['operating_point']['true_positive_rate'] = TPR[index]

		# compute the confidence intervals at the operating point
		n = self.ROC_curve['number_of_negatives']
		x = self.ROC_curve['operating_point']['false_positive_rate']*n
		#self.ROC_curve['operating_point']['false_positive_rate_CI'] = CI_binomial_proportion(x,n)
		
		n = self.ROC_curve['number_of_positives']
		x = self.ROC_curve['operating_point']['true_positive_rate']*n
		#self.ROC_curve['operating_point']['true_positive_rate_CI'] = CI_binomial_proportion(x,n)
		# compute the accuracy
		num1 = self.ROC_curve['operating_point']['true_positive_rate']*self.ROC_curve['number_of_positives']
		num2 = (1-self.ROC_curve['operating_point']['false_positive_rate'])*self.ROC_curve['number_of_negatives']
		num  = num1 + num2
		den  = self.ROC_curve['number_of_instances']
		self.ROC_curve['operating_point']['accuracy'] = num/den
		#self.ROC_curve['operating_point']['accuracy_CI'] = CI_binomial_proportion(num,den)		

	def plot(self, ROC_curve = None,
		line_color    = "black",
		line_width    = 2.0,
		line_style    = "-",
		x_lower_limit = 0.0,
		x_upper_limit = 1.0,
		y_lower_limit = 0.0,
		y_upper_limit = 1.0,
		x_label       = None,
		y_label       = None,
		font_size     = 14,
		label_text    = None,
		ROC_label     = 'Metrics', 
		show_operating_point = True, 
		use_existing_figure_number = False):
		"""plots the ROC curve
		"""
		if ROC_curve is None:
			ROC_curve = self.ROC_curve

		if label_text is None:
			if 'AUC_CI' in ROC_curve:
				label_text = '%s\nAUC = %1.2f [%1.2f, %1.2f]'%(ROC_label,
					ROC_curve['AUC'],
					ROC_curve['AUC_CI']['lower_limit'],
					ROC_curve['AUC_CI']['upper_limit'])
			else:
				label_text = '%s\nAUC = %1.2f'%(ROC_label,
					ROC_curve['AUC'])

		if use_existing_figure_number:
			figure_number = plt.figure(num = use_existing_figure_number)
		else:
			figure_number = plt.figure()

		# plot the operating point
		if show_operating_point:
			if 'operating_point' in ROC_curve:

				x = ROC_curve['operating_point']['false_positive_rate']
				y = ROC_curve['operating_point']['true_positive_rate']
				plt.scatter(x, y, 50, color = line_color)

				plt.annotate('%1.2f @ %1.2f'%(y,x),
					xy = (x, y), 
					xycoords = 'data',
					xytext = (+10, +30),
					textcoords = 'offset points', 
					fontsize = font_size,
					arrowprops = dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

				if 'false_positive_rate_CI' in ROC_curve['operating_point']:
					x_begin = ROC_curve['operating_point']['false_positive_rate_CI']['lower_limit']
					x_end   = ROC_curve['operating_point']['false_positive_rate_CI']['upper_limit']
					y_begin = ROC_curve['operating_point']['true_positive_rate']
					y_end   = ROC_curve['operating_point']['true_positive_rate']
					plt.plot([x_begin, x_end], [y_begin, y_end],
						color = line_color,
						alpha = 0.25,
						linewidth = 1.0)

				if 'true_positive_rate_CI' in ROC_curve['operating_point']:
					y_begin = ROC_curve['operating_point']['true_positive_rate_CI']['lower_limit']
					y_end   = ROC_curve['operating_point']['true_positive_rate_CI']['upper_limit']
					x_begin = ROC_curve['operating_point']['false_positive_rate']
					x_end   = ROC_curve['operating_point']['false_positive_rate']
					plt.plot([x_begin, x_end], [y_begin, y_end],
						color = line_color,
						alpha = 0.25,
						linewidth = 1.0)	

				label_text = '%s\nTPR = %1.2f [%1.2f, %1.2f]\nFPR = %1.2f [%1.2f, %1.2f]\nACC = %1.2f [%1.2f, %1.2f]'%(label_text,
					ROC_curve['operating_point']['true_positive_rate'],
					ROC_curve['operating_point']['true_positive_rate_CI']['lower_limit'],
					ROC_curve['operating_point']['true_positive_rate_CI']['upper_limit'],
					ROC_curve['operating_point']['false_positive_rate'],
					ROC_curve['operating_point']['false_positive_rate_CI']['lower_limit'],
					ROC_curve['operating_point']['false_positive_rate_CI']['upper_limit'],
					ROC_curve['operating_point']['accuracy'],
					ROC_curve['operating_point']['accuracy_CI']['lower_limit'],
					ROC_curve['operating_point']['accuracy_CI']['upper_limit'])

		plt.plot(ROC_curve['false_positive_rate'], ROC_curve['true_positive_rate'],
			color = line_color, 
			linewidth = line_width, 
			linestyle = line_style,
			label = label_text)

		plt.xlim(x_lower_limit,x_upper_limit)
		plt.ylim(y_lower_limit,y_upper_limit)
		if x_label is None:
			x_label = 'False Positive Rate (%d negatives)'%(ROC_curve['number_of_negatives'])
		if y_label is None:
			y_label = 'True Positive Rate (%d positives)'%(ROC_curve['number_of_positives'])
		plt.xlabel(x_label, fontsize = font_size)
		plt.ylabel(y_label, fontsize = font_size)
		plt.legend(loc='best')
		plt.show()	

		return figure_number

	def compute_AUC_CI(self, y_score, y_true, 
		alpha = 0.05,
		method = 'bootstrap',
		number_of_bootstrap_replications = 100):
		""" compute the (1-alpha) confidence intervals for AUC based on bootstrapping
		"""
		CI = {}
		CI['bootstrap_replicates'] = []
		n = len(y_score)
		for i in xrange(number_of_bootstrap_replications):
			bootstrap_index = randint(0,n,n)
			y_score_bootstrap_replicate = [y_score[index] for index in bootstrap_index]
			y_true_bootstrap_replicate  = [y_true[index] for index in bootstrap_index]
			ROC_curve_bootstrap_replicate = self.compute_ROC(y_score_bootstrap_replicate, y_true_bootstrap_replicate)
			CI['bootstrap_replicates'].append(ROC_curve_bootstrap_replicate['AUC'])
		CI['AUC'] = mean(CI['bootstrap_replicates'])
		CI['alpha'] = alpha
		CI['method'] = method
		CI['lower_limit'] = percentile(CI['bootstrap_replicates'], (alpha/2.0)*100)
		CI['upper_limit'] = percentile(CI['bootstrap_replicates'], (1-(alpha/2.0))*100)

		return CI
		
def ROCCurve_example():
	y_score = [0.1, 0.57, 0.3, 0.65, 0.1, 0.0, 0.8, 0.9, 0.1, 0.1, 0.01, 0.2, 0.05]
	y_true  = [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]

	ROC = ROCCurve(y_score, y_true)
	ROC.set_operating_point(operating_point_threshold = 0.5)
	ROC.plot()

    # If you want to overlay plots    
	ROC.set_operating_point(operating_point_FPR = 0.4)
	ROC.plot(ROC_label = 'operating point 1',
		line_color = "red", 
		use_existing_figure_number = 2)

	ROC.set_operating_point(operating_point_TPR = 0.8)
	ROC.plot(ROC_label = 'operating point 2',
		line_color = "blue", 
		use_existing_figure_number = 2)

if __name__=='__main__':
    ROCCurve_example()

