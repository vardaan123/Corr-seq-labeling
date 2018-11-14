#!/usr/bin/python
# -*- coding: utf-8 -*-
""" string cleaning 
"""

import re

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["clean_str","clean_str_teddata"]

def clean_str(string, lower=True):
    """string cleaning 
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if lower:
        string = string.lower()
    string = string.strip()    

    return string

def clean_str_teddata(string, lower=True):
    """string cleaning 
    """
    string = re.sub(r"</s>", " ", string)
    if lower:
        string = string.lower()
    string = string.strip()    

    return string


    
if __name__ == '__main__':
    string = r'Laws such as the Official Secrets Act and Prevention of Terrorist Activities Act[REF]'
    print string
    print clean_str(string)
