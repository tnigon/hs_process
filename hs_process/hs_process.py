# -*- coding: utf-8 -*-
from hyperspectral import IO_tools
from hyperspectral import HS_tools


class HS_process(object):
    '''
    Parent class for batch processing hyperspectral image data
    '''
    def __init__(self, base_dir=None, fname=None):
        '''
        User can base either `base_dir` (`str`) or fname (`str`). `base_dir`
        take precedence over `fname` both are not `None`.
        '''
        self.base_dir = base_dir
        self.fname = fname

        self.fname_list = None
        self.img_spy = None

        def read_inputs(base_dir=self.base_dir, fname=self.fname):
            '''
            Checks `base_dir` and `fname` to determine which to load into this
            class at this point.
            '''
            if base_dir is not None:
                self.fname_list = IO_tools._recurs_dir(self.base_dir)
            elif base_dir is None and fname is not None:
                self.img_spy = IO_tools.read_cube(fname)

        read_inputs()


base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-29_AERF-plot2'

hs = HS_process(base_dir)
