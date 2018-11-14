""" confidence interval for binomial proportion
"""

from scipy.stats import beta

__all__ = ["CI_binomial_proportion","CI_binomial_proportion_example"]

def CI_binomial_proportion(x, n, alpha = 0.05, method = 'Jeffreys'):
    """ Confidence Intervals for the binomial proportion

    Computes the confidence interval CI) for the probability of sucess (p=x/n) 
    in a binomial distribution such that Pr(p belongs to CI) = 1-alpha
        
    Parameters:
    ----------
    x : the numerator   (the number of suceesses)
    n : the denominator (the number of trials)
    alpha : the confidence level (between 0 and 1) [DEFAULT is 0.05]
    method : the method to use [DEFAULT is 'Jeffreys']
    'Jeffreys' -- Equal tailed with non-informative Jeffreys prior

    [not yet implemented]
    'AgrestiCoull' --  same form as Wald but add 2 sucess and 2 failures
    'Wilson' -- as good as Jeffreys 
    'Wald' -- standard normal approximation

    Returns:
    -------
    CI['lower_limit'] : the lower limit of the confidence interval
    CI['lower_limit'] : the upper limit of the confidence interval

    Reference:
    ----------

    Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. 
    Interval Estimation for a Binomial Proportion
    Statistical Science 16 (2001): 101-133.

    The commonly  used standard interval based on the normal approximation
    (also known as the Wald interval) is not recommended since it performs
    poorly for small sample sizes and near the endpoints and more importantly 
    is persistently chaotic (see (Brown, Cai and DasGupta 2001) for 
    illustrations of this phenomenon. To quote the authors 'The performance
    is so erratic and the qualifications given in the influential texts are 
    so defective that the standard interval should not be used').  Because of 
    the discreteness of the binomial distribution the exact Clopper-Pearson
    interval is overly conservative and is also not recommended.  

    """
    CI = {}
    
    CI['x'] = x
    CI['n'] = n
    CI['binomial_proportion'] = x/float(n)
    
    CI['alpha'] = alpha
    CI['method'] = method
    if method == 'Jeffreys':
        CI['lower_limit'] = beta.ppf(alpha/2.0,x+0.5,n-x+0.5)
        CI['upper_limit'] = beta.ppf(1-(alpha/2.0),x+0.5,n-x+0.5)
  
    return CI

def CI_binomial_proportion_example():
    
    CI = CI_binomial_proportion(45, 50)

    CI = CI_binomial_proportion(49, 50, alpha = 0.05, method = 'Jeffreys')

    print(CI)

    return CI

if __name__=='__main__':
    CI_binomial_proportion_example()
