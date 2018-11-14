""" Bunch pattern
"""

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["Bunch"]

class Bunch(object):
  def __init__(self, dict):
    self.__dict__.update(dict)


