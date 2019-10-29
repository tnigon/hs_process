# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:23:32 2019

@author: nigo0024
"""

class band_selection(object):
    '''
    Class for performing band selection on hyperspectral data. Both the spectra
    and the "ground truth" data to be predicted are required for this
    step.
    '''
    def __init__(self, base_dir=None, search_ext='.bip', dir_level=0):