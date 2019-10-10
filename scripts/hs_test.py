# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:41:15 2019

@author: nigo0024
"""

from hs_process import hsp
from hs_process import hstools
from hs_process import hsio

base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-29_AERF-plot2'
hs = hsp(base_dir,search_exp='.bip', recurs_level=0)
fname = hs.fname_list[0]
io = hsio(fname)
io = hsio()
array = io.img_sp.load()

tools = hstools(io.img_sp)
meta_bands = tools.get_meta_bands()


meta_bands = io.meta_bands
