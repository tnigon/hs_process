# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:41:15 2019

@author: nigo0024
"""
from hs_process.hs_process import HS_process
from hs_process.helper import IO_tools

base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-29_AERF-plot2'

hs = HS_process(base_dir,search_exp='.bip', recurs_level=0)
hs.base_dir
hs.fname_list
fname = hs.fname_list[0]
img_spy = IO_tools.read_cube(fname)
