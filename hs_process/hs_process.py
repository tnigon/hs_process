# -*- coding: utf-8 -*-
import os

from hs_process.helper import IO_tools
#from hyperspectral import HS_tools


class HS_process(object):
    '''
    Parent class for batch processing hyperspectral image data
    '''
    def __init__(self, base_dir=None, fname=None, search_exp='.bip',
                 recurs_level=0):
        '''
        User can base either `base_dir` (`str`) or fname (`str`). `base_dir`
        take precedence over `fname` both are not `None`.
        '''
        self.base_dir = base_dir
        self.fname = fname
        self.search_exp = search_exp
        self.recurs_level = recurs_level

        self.fname_list = None
        self.img_spy = None

        def read_inputs():
            '''
            Checks `base_dir` and `fname` to determine which to load into this
            class at this point.
            '''
            if self.base_dir is not None:
                self.fname_list = self._recurs_dir(self.base_dir,
                                                  self.search_exp,
                                                  self.recurs_level)
            elif self.base_dir is None and self.fname is not None:
                self.img_spy = IO_tools.read_cube(fname)

        read_inputs()

    def _recurs_dir(self, base_dir, search_exp='.csv', level=None):
        '''
        Searches all folders and subfolders recursively within <base_dir>
        for filetypes of <search_exp>.
        Returns sorted <outFiles>, a list of full path strings of each result.

        Inputs:     Directory (base_dir) that includes files
                    (may be in subdirectories)
            base_dir: directory path that should include files to be returned
            search_exp: file format to search for in all directories
                    and subdirectories
                    Note: may include files with <search_exp> in name that have
                    different extensions.
            level: how many levels to search; if None, searches all levels
        Outputs:    out_files include the full pathname\filename\ext of all
                    files that have <search_exp> in their name.
        ===================================================================
        '''
        if level is None:
            level = 1
        else:
            level -= 1
        d_str = os.listdir(base_dir)
        out_files = []
        for item in d_str:
            full_path = os.path.join(base_dir, item)
            if not os.path.isdir(full_path) and item.endswith(search_exp):
                out_files.append(full_path)
            elif os.path.isdir(full_path) and level >= 0:
                new_dir = full_path  # If dir, then search in that
                out_files_temp = self._recurs_dir(new_dir, search_exp)
                if out_files_temp:  # if list is not empty
                    out_files.extend(out_files_temp)  # add items
        return sorted(out_files)

