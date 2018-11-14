""" small utility to visualize the feature names and their weights
"""

import matplotlib.pyplot as plt
import numpy as np

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["plot_feature_weights","plot_feature_weights_debater"]

def plot_feature_weights(feature_names, feature_weights, 
	number_of_features_to_display = None,
	face_color = '#ff9999',
	y_label = 'feature weights',
	figure_name = 'feature weights',):
	""" plot the feature weights in the decreasing order along with the feature names

	:params:
	    feature_names : list of strings
	    	the feature names
	    feature_weights : list of floats
	        the corresponding weights
        number_of_features_to_display : int
            the number of features to plot
            (default: all features)

	Example usage:
	--------------
	>>> feature_names   = ['BLEU', 'NIST', 'synonym', 'junk']
	>>> feature_weights = [-1.7, 2.3, 0.6, -0.1]
	>>> plot_feature_weights(feature_names, feature_weights)
	
	"""

	number_of_features = len(feature_names)
	if number_of_features_to_display is None:
		number_of_features_to_display = number_of_features
	else:
		number_of_features_to_display = min(number_of_features_to_display, number_of_features)

	format_width = max([len(name) for name in feature_names])

	sorted_features = sorted(zip(feature_names,feature_weights), 
		key = (lambda x:abs(x[1])), reverse = True)

	print('Top %d[/%d] discriminative features '%(number_of_features_to_display,number_of_features))
	print('%s : %s'%('feature name'.rjust(format_width+1),'feature weight'))
	y_axis = []
	tick_labels = []
	for i in xrange(number_of_features_to_display):
		f_name   = sorted_features[i][0]
		f_weight = sorted_features[i][1]
		print('%s : %f'%(f_name.rjust(format_width+1),f_weight))
		y_axis.append(f_weight)
		tick_labels.append(f_name)
	
	width  = y_axis
	bottom = np.arange(len(y_axis)) + 0.5
	
	plt.figure(num = figure_name)
	plt.bar(bottom, width, 
    	align = 'center',
    	facecolor = face_color,
    	edgecolor = 'white')
	plt.ylabel(y_label)
	plt.xticks(bottom, [])
	for i in xrange(number_of_features_to_display):
		plt.text(bottom[i],0.0,tick_labels[i], 
    		rotation = 'vertical',
    		verticalalignment='center')
	plt.show()

def plot_feature_weights_debater(model_filename,
	number_of_features_to_display = None,
	max_feature_name_length = 50):
	""" visualize the debater models

    :params:
        model_filename : str
            the model filename
        number_of_features_to_display : int
            the number of features to plot
            (default: all features)
        max_feature_name_length : int
        	truncate the feature names since they tend to be very long 

    :sample model file:

    LR
    Model:
    Free:-2.998142006684309
    TenseMajority.CLAIM_ONLY:0.12913853218847698
    PatternCount[Pattern=0-9, Index=0, Normalize=true].CLAIM_ONLY:-2.840261470998678
    PatternCount[Pattern=A-Z, Index=0, Normalize=true].CLAIM_ONLY:-1.138288341844977
    PatternCount[Pattern=DOT, Index=0, Normalize=true].CLAIM_ONLY:-0.3940854734644555
    PatternCount[Pattern=COMMA, Index=0, Normalize=true].CLAIM_ONLY:-0.5653563606986576
    PatternCount[Pattern=LEFT_PARENTHRIGHT_PARENTH, Index=0, Normalize=true].CLAIM_ONLY:-0.5096724456685616
    PatternCount[Pattern=QUOTE, Index=0, Normalize=true].CLAIM_ONLY:0.7090667621695694
    PatternCount[Pattern=REF, Index=0, Normalize=true].CLAIM_ONLY:-0.46032636582704095
    SentenceLength[Tokenizer=OpenNlpTokenizer, Sign=false].CLAIM_ONLY:-1.0247762471074533    	
  
	"""
	f = open(model_filename,'r')
	
	name = f.readline()
	model_name = f.readline()
	bias = f.readline()

	feature_names = []
	feature_weights = []
	for line in f:
		temp=line.strip().split(':')
		feature_names.append(':'.join(temp[:-1])[:max_feature_name_length])
		feature_weights.append(float(temp[-1]))
	
	f.close()

	plot_feature_weights(feature_names, feature_weights, 
		number_of_features_to_display=number_of_features_to_display)

if __name__ == '__main__':
	""" example usage
	"""
	plot_feature_weights_debater(model_filename = 'model.txt',
		number_of_features_to_display = None,
		max_feature_name_length = 50)

	