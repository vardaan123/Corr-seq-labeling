""" Precision Recall Curve
"""
import matplotlib.pyplot as plt
#from CI_binomial_proportion import *

__all__ = ["PRCurve"]

class PRCurve():
    """ Precision Recall Curve
    """
    def __init__(self, y_score, y_true):
        self.PR_curve = self.compute_PR_curve(y_score, y_true)

    def compute_PR_curve(self, y_score, y_true):
        """ computes the PR Curve

        Parameters:
        -----------
        y_score : predictions of the classifier which are real numbers
                : larger value indicates higher confidence that the label is 1
        y_true  : the true binary class label [1 is considered positive, rest negtive]

        Returns:
        --------
        PR_curve['recall']   : the X-axis of the PR Curve [sensitivity]
                             : Pr[ y_predicted = 1 | y_true = 1 ]  
                             : Pr[ y_score >= threshold | y_true = 1 ]  
        PR_curve['precision']: the Y-axis of the PR Curve [positive predictive value]
                             : Pr[ y_true = 1 | y_predicted = 1]  
                             : Pr[ y_true = 1 | y_score >= threshold ]  
        PR_curve['threshold']: the correponding thresholds
        PR_curve['average_precision'] : the area under the PR curve

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

        PR_curve = {}
        PR_curve['threshold']           = thresholds
        PR_curve['number_of_positives'] = number_of_positives
        PR_curve['number_of_negatives'] = number_of_negatives
        PR_curve['number_of_instances'] = number_of_positives + number_of_negatives
        
        PR_curve['precision'] = []
        PR_curve['recall']    = []
        PR_curve['number_of_predicted_positives'] = []

        PR_curve['precision'].append(1.0)
        PR_curve['recall'].append(0.0)
        PR_curve['number_of_predicted_positives'].append(0.0)

        for i in xrange(1,len(thresholds)):
            TP = float(number_of_true_positives[i])
            FP = float(number_of_false_positives[i])
            PR_curve['precision'].append(TP/(TP+FP)) 
            PR_curve['recall'].append(TP/number_of_positives) 
            PR_curve['number_of_predicted_positives'].append(TP+FP)

        PR_curve['average_precision'] = self.compute_average_precision(PR_curve)    

        return PR_curve 
    
    def compute_average_precision(self, PR_curve = None, 
        recall_low = 0.0, recall_high = 1.0):
        """ computes the average precision which is essentially the area under 
        the PR curve between precision_low to precision_high

        Parameters:
        -----------
        precision_low  : defaults to 0.0 
        precision_high : defaults to 1.0

        Results:
        --------
        average_precision : the area under the PR curve

        """     
        if PR_curve is None:
            PR_curve = self.PR_curve

        recall    = PR_curve['recall']
        precision = PR_curve['precision']

        temp = [abs(f-recall_low) for f in recall]
        index_low = temp.index(min(temp))

        temp = [abs(f-recall_high) for f in recall]
        temp.reverse()
        index_high = len(temp)-1-temp.index(min(temp))

        average_precision = 0.0
        for i in xrange(index_low+1, index_high+1):
            base   = recall[i]-recall[i-1]
            height = precision[i]
            average_precision   += (base*height)

        return average_precision    

    def set_operating_point(self, 
        operating_point_precision = None,
        operating_point_recall    = None,
        operating_point_threshold = None):
        """ set the operating point
    
        Parameters:
        -----------
        operating_point_precision : target precision
        OR
        operating_point_recall    : target recall
        OR
        operating_point_threshold : target threshold

        Results:
        --------
        self.PR_curve['operating_point']

        """

        precision   = self.PR_curve['precision']
        recall      = self.PR_curve['recall']
        threshold   = self.PR_curve['threshold']
        number_of_predicted_positives = self.PR_curve['number_of_predicted_positives']
     
        if operating_point_precision:
            index = 0
            while precision[index] > operating_point_precision:
                index += 1

        if operating_point_recall:
            index = 0
            while recall[index] < operating_point_recall:
                index += 1

        if operating_point_threshold:
            index = 1
            while threshold[index] > operating_point_threshold:
                index += 1
        
        self.PR_curve['operating_point'] = {}
        self.PR_curve['operating_point']['threshold'] = threshold[index]
        self.PR_curve['operating_point']['precision'] = precision[index]
        self.PR_curve['operating_point']['recall']    = recall[index]
        self.PR_curve['operating_point']['number_of_predicted_positives'] = number_of_predicted_positives[index]
                
        # compute the confidence intervals at the operating point
        n = self.PR_curve['operating_point']['number_of_predicted_positives']
        x = self.PR_curve['operating_point']['precision']*n
        #self.PR_curve['operating_point']['precision_CI'] = CI_binomial_proportion(x,n)
        
        n = self.PR_curve['number_of_positives']
        x = self.PR_curve['operating_point']['recall']*n
        #self.PR_curve['operating_point']['recall_CI'] = CI_binomial_proportion(x,n)

        # compute the F_measure
        P = self.PR_curve['operating_point']['precision']
        R = self.PR_curve['operating_point']['recall']
        self.PR_curve['operating_point']['F_measure'] = self.compute_F_measure(beta = 1.0)

    def compute_F_measure(self, beta = 1.0):
        """ compute the F-measure at the operating point
        weighted harmonic mean of precision and recall
        beta = 1.0 corresponds to the balanced F-measure
        """
        P = self.PR_curve['operating_point']['precision']
        R = self.PR_curve['operating_point']['recall']
        num = ((beta*beta)+1)*P*R
        den = (beta*beta*P)+R
        if den == 0.0:
            F_measure = 0.0
        else:
            F_measure = num/den
        return F_measure

    def plot(self, PR_curve = None,
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
        PR_label      = 'Metrics', 
        show_operating_point = True, 
        use_existing_figure_number = False):
        """plots the PR curve
        """
        if PR_curve is None:
            PR_curve = self.PR_curve

        if label_text is None:
            if 'average_precision_CI' in PR_curve:
                label_text = '%s\nAveP = %1.2f [%1.2f, %1.2f]'%(PR_label, PR_curve['average_precision'],
                    PR_curve['average_precision_CI']['lower_limit'],
                    PR_curve['average_precision_CI']['upper_limit'])
            else:
                label_text = '%s\nAveP = %1.2f'%(PR_label, PR_curve['average_precision'])

        if use_existing_figure_number:
            figure_number = plt.figure(num = use_existing_figure_number)
        else:
            figure_number = plt.figure()

        # plot the operating point
        if show_operating_point:
            if 'operating_point' in PR_curve:

                x = PR_curve['operating_point']['recall']
                y = PR_curve['operating_point']['precision']
                plt.scatter(x, y, 50, color = line_color)

                plt.annotate('%1.2f @ %1.2f'%(y,x),
                    xy = (x, y), 
                    xycoords = 'data',
                    xytext = (+10, +30),
                    textcoords = 'offset points', 
                    fontsize = font_size,
                    arrowprops = dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

                if 'recall_CI' in PR_curve['operating_point']:
                    x_begin = PR_curve['operating_point']['recall_CI']['lower_limit']
                    x_end   = PR_curve['operating_point']['recall_CI']['upper_limit']
                    y_begin = PR_curve['operating_point']['precision']
                    y_end   = PR_curve['operating_point']['precision']
                    plt.plot([x_begin, x_end], [y_begin, y_end],
                        color = line_color,
                        alpha = 0.25,
                        linewidth = 1.0)

                if 'precision_CI' in PR_curve['operating_point']:
                    y_begin = PR_curve['operating_point']['precision_CI']['lower_limit']
                    y_end   = PR_curve['operating_point']['precision_CI']['upper_limit']
                    x_begin = PR_curve['operating_point']['recall']
                    x_end   = PR_curve['operating_point']['recall']
                    plt.plot([x_begin, x_end], [y_begin, y_end],
                        color = line_color,
                        alpha = 0.25,
                        linewidth = 1.0)    

                label_text = '%s\nPrecision = %1.2f [%1.2f, %1.2f]\nRecall      = %1.2f [%1.2f, %1.2f]\nF-score = %1.2f '%(label_text,
                    PR_curve['operating_point']['precision'],
                    PR_curve['operating_point']['precision_CI']['lower_limit'],
                    PR_curve['operating_point']['precision_CI']['upper_limit'],
                    PR_curve['operating_point']['recall'],
                    PR_curve['operating_point']['recall_CI']['lower_limit'],
                    PR_curve['operating_point']['recall_CI']['upper_limit'],
                    PR_curve['operating_point']['F_measure'])

        plt.plot(PR_curve['recall'], PR_curve['precision'],
            color = line_color, 
            linewidth = line_width, 
            linestyle = line_style,
            label = label_text)

        plt.xlim(x_lower_limit,x_upper_limit)
        plt.ylim(y_lower_limit,y_upper_limit)
        if x_label is None:
            x_label = 'Recall (%d positives)'%(PR_curve['number_of_positives'])
        if y_label is None:
            y_label = 'Precision'
        plt.xlabel(x_label, fontsize = font_size)
        plt.ylabel(y_label, fontsize = font_size)
        plt.legend(loc='best')
        plt.show()  

        return figure_number

def PRCurve_example():
    y_score = [0.1, 0.57, 0.3, 0.65, 0.1, 0.0, 0.8, 0.9, 0.1, 0.1, 0.01, 0.2, 0.05]
    y_true  = [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]

    # compute PR curve and plot
    PR = PRCurve(y_score, y_true)
    PR.set_operating_point(operating_point_threshold = 0.5)
    PR.plot()
 
    # If you want to overlay plots    
    PR.set_operating_point(operating_point_recall = 0.5)
    PR.plot(PR_label = 'operating point 1',
        line_color = "red", 
        use_existing_figure_number = 2)

    PR.set_operating_point(operating_point_precision = 0.8)
    PR.plot(PR_label = 'operating point 2',
        line_color = "blue", 
        use_existing_figure_number = 2)

if __name__=='__main__':
    PRCurve_example()

