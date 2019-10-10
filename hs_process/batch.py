# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd

from hs_process import hsio
from hs_process import spec_mod
#from hs_process import Spatial_mod
#from hs_process import Segment


class batch(object):
    '''
    Parent class for batch processing hyperspectral image data
    '''
    def __init__(self, base_dir=None, search_ext='.bip', recurs_level=0):
        '''
        User can pass either `base_dir` (`str`) or fname (`str`). `base_dir`
        takes precedence over `fname` if both are passed (not `None`).
        '''
        self.base_dir = base_dir
        self.fname = fname
        self.search_exp = search_exp
        self.recurs_level = recurs_level

        self.fname_list = None
        self.img_spy = None

        self.io = hsio()

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
                self.img_spy = hsio.read_cube(fname)

        read_inputs()

    def _execute_spec_clip(self, fname_list, base_dir_out, folder_name,
                           name_append, wl_bands):
        '''
        Actually executes the spectral clip to keep the main function a bit
        cleaner
        '''
        for fname in fname_list:
            print('Spectrally clipping: {0}\n'.format(fname))
            self.io.read_cube(fname)  # options: name_long, name_plot, name_short, individual_plot, overwrite
            sm = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out = self._save_file_setup(base_dir, folder_name)
            else:
                dir_out = self._save_file_setup(base_dir_out, folder_name)
            if self.io.name_plot is not None:
                name_print = self.io.name_plot
            else:
                name_print = self.io.name_short
            if name_append is None:
                name_append = ''

            array_clip, metadata = sm.spectral_clip(wl_bands=wl_bands)

            metadata['interleave'] = self.io.defaults.interleave
            name_label = (name_print + '-' + str(name_append) + '.' +
                          self.io.defaults.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_clip,
                               dtype=self.io.defaults.dtype,
                               force=self.io.defaults.force,
                               ext=self.io.defaults.ext,
                               interleave=self.io.defaults.interleave,
                               byteorder=self.io.defaults.byteorder,
                               metadata=metadata)

    def _execute_spec_smooth(self, fname_list, base_dir_out, folder_name,
                             name_append, window_size, order, stats):
        '''
        Actually executes the spectral smooth to keep the main function a bit
        cleaner
        '''
        if stats is True:
            df_smooth_stats = pd.DataFrame(columns=['fname', 'mean', 'std', 'cv'])

        for fname in fname_list:
            print('Spectrally smoothing: {0}\n'.format(fname))
            self.io.read_cube(fname)  # options: name_long, name_plot, name_short, individual_plot, overwrite
            sm = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out = self._save_file_setup(base_dir, folder_name)
            else:
                dir_out = self._save_file_setup(base_dir_out, folder_name)
            if self.io.name_plot is not None:
                name_print = self.io.name_plot
            else:
                name_print = self.io.name_short
            if name_append is None:
                name_append = ''

            array_smooth, metadata = sm.spectral_smooth(
                    window_size=window_size, order=order)

            metadata['interleave'] = self.io.defaults.interleave
            name_label = (name_print + '-' + str(name_append) + '.' +
                          self.io.defaults.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_smooth,
                               dtype=self.io.defaults.dtype,
                               force=self.io.defaults.force,
                               ext=self.io.defaults.ext,
                               interleave=self.io.defaults.interleave,
                               byteorder=self.io.defaults.byteorder,
                               metadata=metadata)

            if stats is True:
                mean = np.nanmean(array_smooth)
                std = np.nanstd(array_smooth)
                cv = std/mean
                df_smooth_temp = pd.DataFrame([[fname, mean, std, cv]],
                                              columns=['fname', 'mean', 'std',
                                                       'cv'])
                df_smooth_stats = df_smooth_stats.append(df_smooth_temp,
                                                         ignore_index=True)
        if stats is True:
            fname_stats = os.path.join(base_dir_out, name_append +
                                       '_stats.csv')
            df_smooth_stats.to_csv(fname_stats)
            return df_smooth_stats

    def _recurs_dir(self, base_dir, search_ext='.csv', level=None):
        '''
        Searches all folders and subfolders recursively within <base_dir>
        for filetypes of <search_exp>.
        Returns sorted <outFiles>, a list of full path strings of each result.

        Parameters:
            base_dir: directory path that should include files to be returned
            search_ext: file format/extension to search for in all directories
                and subdirectories
            level: how many levels to search; if None, searches all levels

        Returns:
            out_files: include the full pathname\filename\ext of all files that
                have <search_exp> in their name.
        '''
        if level is None:
            level = 1
        else:
            level -= 1
        d_str = os.listdir(base_dir)
        out_files = []
        for item in d_str:
            full_path = os.path.join(base_dir, item)
            if not os.path.isdir(full_path) and item.endswith(search_ext):
                out_files.append(full_path)
            elif os.path.isdir(full_path) and level >= 0:
                new_dir = full_path  # If dir, then search in that
                out_files_temp = self._recurs_dir(new_dir, search_ext)
                if out_files_temp:  # if list is not empty
                    out_files.extend(out_files_temp)  # add items
        return sorted(out_files)

    def _save_file_setup(self, base_dir_out, folder_name, fname=None):
        '''
        Basic setup items when saving manipulated image files to disk

        Parameters:
            base_dir_out (`str`): Parent directory that all processed datacubes
                will be saved.
            folder_name (`str`): Folder to add to `base_dir_out` to save all
                the processed datacubes.
            fname (`str`):
        '''
#        if base_dir_out is None:
#            base_dir_out = os.path.join(self.base_dir, folder_name)
        dir_out = os.path.join(base_dir_out, folder_name)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)

        if fname is not None:
            name_print = self.io.name_short
        else:
            name_print = self.name_short
        return base_dir_out, name_print
        return dir_out

    def spectral_clip(self, fname_list=None, base_dir=None, search_ext='bip',
                      dir_level=0, base_dir_out=None, folder_name='spec_clip',
                      name_append='spec-clip',
                      wl_bands=[[0, 420], [760, 776], [813, 827], [880, 1000]],
                      out_dtype=False, out_force=None, out_ext=False,
                      out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to spectrally clip multiple datacubes in the same
        way.

        Parameters:
            fname_list (`list`, optional): list of filenames to process; if
                left to `None`, will look at `base_dir`, `search_ext`, and
                `dir_level` parameters for files to process (default: `None`).
            base_dir (`str`, optional): directory path to search for files to
                spectrally clip; if `fname_list` is not `None`, `base_dir` will
                be ignored (default: `None`).
            search_ext (`str`): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if `fname_list` is not `None`, `search_ext` will
                be ignored (default: 'bip').
            dir_level (`int`): The number of directory levels to search; if
                `None`, searches all directory levels (default: 0).
            base_dir_out (`str`): directory path to save all processed
                datacubes; if set to `None`, a folder named according to the
                `folder_name` parameter is added to `base_dir`
            folder_name (`str`): folder to add to `base_dir_out` to save all
                the processed datacubes (default: 'spec-clip').
            name_append (`str`): name to append to the filename (default:
                'spec-clip').
            wl_bands (`list` or `list of lists`): minimum and maximum
                wavelenths to clip from image; if multiple groups of
                wavelengths should be cut, this should be a list of lists. For
                example, wl_bands=[760, 776] will clip all bands greater than
                760.0 nm and less than 776.0 nm;
                wl_bands = [[0, 420], [760, 776], [813, 827], [880, 1000]]
                will clip all band less than 420.0 nm, bands greater than 760.0
                nm and less than 776.0 nm, bands greater than 813.0 nm and less
                than 827.0 nm, and bands greater than 880 nm (default).
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in `batch.io.defaults, and are
                therefore accessible at a high level. See
                `hsio.set_io_defaults()` for more information on each of the
                settings.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is not None:
            self._execute_spec_clip(fname_list, base_dir_out, folder_name,
                                    name_append, wl_bands)
        elif base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_spec_clip(fname_list, base_dir_out, folder_name,
                                    name_append, wl_bands)
        else:  # fname_list and base_dir are both `None`
            base_dir = self.base_dir  # base_dir may have been stored to the `batch` object
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_spec_clip(fname_list, base_dir_out, folder_name,
                                    name_append, wl_bands)

    def spectral_smooth(self, fname_list=None, base_dir=None, search_ext='bip',
                        dir_level=0, base_dir_out=None,
                        folder_name='spec_smooth', name_append='spec-smooth',
                        window_size=11, order=2, stats=False,
                        out_dtype=False, out_force=None, out_ext=False,
                        out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to spectrally smooth multiple datacubes in the
        same way.

        Parameters:
            fname_list (`list`, optional): list of filenames to process; if
                left to `None`, will look at `base_dir`, `search_ext`, and
                `dir_level` parameters for files to process (default: `None`).
            base_dir (`str`, optional): directory path to search for files to
                spectrally clip; if `fname_list` is not `None`, `base_dir` will
                be ignored (default: `None`).
            search_ext (`str`): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if `fname_list` is not `None`, `search_ext` will
                be ignored (default: 'bip').
            dir_level (`int`): The number of directory levels to search; if
                `None`, searches all directory levels (default: 0).
            base_dir_out (`str`): directory path to save all processed
                datacubes; if set to `None`, a folder named according to the
                `folder_name` parameter is added to `base_dir`
            folder_name (`str`): folder to add to `base_dir_out` to save all
                the processed datacubes (default: 'spec-smooth').
            name_append (`str`): name to append to the filename (default:
                'spec-smooth').
            window_size (`int`): the length of the window; must be an odd
                integer number (default: 11).
            order (`int`): the order of the polynomial used in the filtering;
                must be less than `window_size` - 1 (default: 2).
            stats (`bool`): whether to compute some basic descriptive
                statistics (mean, st. dev., and coefficient of variation) of
                the smoothed data array (default: `False`)
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in `batch.io.defaults, and are
                therefore accessible at a high level. See
                `hsio.set_io_defaults()` for more information on each of the
                settings.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is not None:
            self._execute_spec_smooth(fname_list, base_dir_out, folder_name,
                                      name_append, window_size, order)
        elif base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_spec_smooth(fname_list, base_dir_out, folder_name,
                                      name_append, window_size, order)
        else:  # fname_list and base_dir are both `None`
            base_dir = self.base_dir  # base_dir may have been stored to the `batch` object
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_spec_smooth(fname_list, base_dir_out, folder_name,
                                      name_append, window_size, order)
