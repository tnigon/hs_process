# -*- coding: utf-8 -*-
import geopandas as gpd
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import sys
import time
from tqdm import tqdm

from matplotlib import pyplot as plt
import warnings

from hs_process.utilities import defaults
from hs_process.utilities import hsio
from hs_process.segment import segment
from hs_process.spec_mod import spec_mod
from hs_process.spatial_mod import spatial_mod

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


class batch(object):
    '''
    Class for batch processing hyperspectral image data. Makes use of
    `segment`_, `spatial_mod`_, and `spec_mod`_ to batch process many
    datacubes in a given directory. Supports options to save full
    datacubes, geotiff renders, as well as summary statistics and/or
    reports for the various tools.

    Note:
        It may be a good idea to review and understand the `defaults`_,
        `hsio`_, `hstools`_, `segment`_, `spatial_mod`_, and `spec_mod`_
        classes prior to using the ``batch`` module.

    .. _defaults: hs_process.defaults.html
    .. _hsio: hs_process.hsio.html
    .. _hstools: hs_process.hstools.html
    .. _segment: hs_process.segment.html
    .. _spatial_mod: hs_process.spatial_mod.html
    .. _spec_mod: hs_process.spec_mod.html
    '''
    def __init__(self, base_dir=None, search_ext='.bip', dir_level=0,
                 lock=None, progress_bar=False):
        '''
        Parameters:
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            lock (``multiprocessing.Lock``): Can be passed to ensure lock is in
                place when writing to a file during multiprocessing.
        '''
        self.base_dir = base_dir
        self.search_ext = search_ext
        self.dir_level = dir_level
        self.lock = lock
        self.progress_bar = progress_bar

        self.fname_list = None
        if base_dir is not None:
            self.fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        self.io = hsio()
        self.my_spectral_mod = None
        self.my_spatial_mod = None
        self.my_segment = None

    def _try_spat_crop_col_key(self, key, df_row):
        '''
        Gets value of ``key`` (column name) from ``df_row``; returns
        ``None`` if there is a KeyError

        This is tricky for crop_X and buf_X columns, because we must decipher
        whether to get these values from the default pool or not. If we get a
        KeyError, our first instinct is to gather the default, but we must
        check the "inverse" first (the "inverse" of crop_e_pix is crop_e_m) to
        avoid overwriting a value passed in df_row unintentionally. Therefore,
        this function handles keys differently if "crop" or "buf" are part of
        ``key`` than if they are not part of ``key``

        Adds ``key`` to batch.io.defaults.spat_crop_cols if it does not yet
        exist, but then of course the ``value`` that is returned will be
        ``None``
        '''
        if key not in self.io.defaults.spat_crop_cols.keys():
            print('Adding key "{0}" to defaults.spat_crop_cols dictionary'
                  ''.format(key))
            self.io.defaults.spat_crop_cols[key] = key

        try:
            value = df_row[self.io.defaults.spat_crop_cols[key]]
        except KeyError:  # try to retrieve a default value
            # decide whehter to get default or not.. how?
            # check the inverse to see if it is accesible
            # try:
            #     value = self.io.defaults.crop_defaults[key]
            # except KeyError:
            #     value = None
            if 'crop' in key or 'buf' in key:
                key_base = key[:key.find('_', key.rfind('_'))]
                key_unit = key[key.find('_', key.rfind('_')):]
                if key_unit == '_m':
                    key_unit_inv = '_pix'
                elif key_unit == '_pix':
                    key_unit_inv = '_m'
                try:
                    value_inv = df_row[self.io.defaults.spat_crop_cols[key_base+key_unit_inv]]  # exists; set to NaN and carry on
                    value = None
                except KeyError:  # neither exist, gather default
                    try:
                        value = self.io.defaults.crop_defaults[key]
                    except KeyError:
                        value = None
            else:  # proceed as normal
                try:
                    value = self.io.defaults.crop_defaults[key]
                except KeyError:
                    value = None

        # if key in ['crop_e_m', 'crop_n_m', 'crop_e_pix', 'crop_n_pix']:
        #     print('Key: {0}  Value: {1}'.format(key, value))
        return value

    def _check_processed(self, fname_list, base_dir_out, folder_name,
                         name_append, append_extra=None, ext=None):
        '''
        Checks if any files in fname_list have already (presumably) undergone
        processing. This is determined by checking if a file exists with a
        particular name based on the filename in fname_list and naming
        parameters (i.e,. ``folder_name`` and ``name_append``).

        Parameters:
            ext (``str``): e.g., '.spec'
        '''
        if append_extra is None:
            append_extra = ''
        fname_list_final = fname_list.copy()
        for fname in fname_list:
            if base_dir_out is None:
                base_dir = os.path.split(fname)[0]
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print(fname)

            if ext is None:
                name_label = (name_print + name_append + append_extra + '.' +
                              self.io.defaults.envi_write.interleave)
            else:
                name_label = (name_print + name_append + append_extra + ext)
            if os.path.isfile(os.path.join(dir_out, name_label)):
                fname_list_final.remove(fname)
        msg1 = ('There are no files to process. Please check if files have '
                'already undergone processing. If existing files should be '
                'overwritten, be sure to set the ``out_force`` parameter.\n')
        msg2 = ('Processing {0} files. If existing files should be '
                'overwritten, be sure to set the ``out_force`` parameter.\n'
                ''.format(len(fname_list_final)))
        if not len(fname_list_final) > 0:
            warnings.warn(msg1, UserWarning, stacklevel=0)
        else:
            print(msg2)
        time.sleep(0.2)  # when using progress bar, this keeps from splitting lines
        return fname_list_final

    def _crop_read_sheet(self, row):
        '''
        Reads the necessary information from the spreadsheet and saves it
        to a dictionary

        If this function causes an error, try checking
        ``batch.io.defaults.spat_crop_col`` - these should be adjusted
        according to the default column names of the input (i.e.,
        ``fname_sheet``).
        '''
        crop_specs = {
                'directory': self._try_spat_crop_col_key('directory', row),
                'fname': self._try_spat_crop_col_key('fname', row),
                'name_short': self._try_spat_crop_col_key('name_short', row),
                'name_long': self._try_spat_crop_col_key('name_long', row),
                'ext': self._try_spat_crop_col_key('ext', row),
                'pix_e_ul': self._try_spat_crop_col_key('pix_e_ul', row),
                'pix_n_ul': self._try_spat_crop_col_key('pix_n_ul', row),
                'plot_id_ref': self._try_spat_crop_col_key('plot_id_ref', row),
                'alley_size_e_m': self._try_spat_crop_col_key('alley_size_e_m', row),
                'alley_size_n_m': self._try_spat_crop_col_key('alley_size_n_m', row),
                'alley_size_e_pix': self._try_spat_crop_col_key('alley_size_e_pix', row),
                'alley_size_n_pix': self._try_spat_crop_col_key('alley_size_n_pix', row),
                'buf_e_m': self._try_spat_crop_col_key('buf_e_m', row),
                'buf_n_m': self._try_spat_crop_col_key('buf_n_m', row),
                'buf_e_pix': self._try_spat_crop_col_key('buf_e_pix', row),
                'buf_n_pix': self._try_spat_crop_col_key('buf_n_pix', row),
                'crop_e_m': self._try_spat_crop_col_key('crop_e_m', row),
                'crop_n_m': self._try_spat_crop_col_key('crop_n_m', row),
                'crop_e_pix': self._try_spat_crop_col_key('crop_e_pix', row),
                'crop_n_pix': self._try_spat_crop_col_key('crop_n_pix', row),
                'gdf_shft_e_pix': self._try_spat_crop_col_key('gdf_shft_e_pix', row),
                'gdf_shft_n_pix': self._try_spat_crop_col_key('gdf_shft_n_pix', row),
                'gdf_shft_e_m': self._try_spat_crop_col_key('gdf_shft_e_m', row),
                'gdf_shft_n_m': self._try_spat_crop_col_key('gdf_shft_n_m', row),
                'n_plots_x': self._try_spat_crop_col_key('n_plots_x', row),
                'n_plots_y': self._try_spat_crop_col_key('n_plots_y', row),
                'n_plots': self._try_spat_crop_col_key('n_plots', row)}
        if crop_specs['fname'] is None:
            try:
                crop_specs['fname'] = (crop_specs['name_short'] +
                                       crop_specs['name_long'] +
                                       crop_specs['ext'])
            except TypeError:
                crop_specs['fname'] = None
        if crop_specs['fname'] is not None:
            base_name = os.path.basename(crop_specs['fname'])
            if crop_specs['name_short'] is None:
                crop_specs['name_short'] = base_name[
                        :base_name.find('-', base_name.rfind('_'))]
            if crop_specs['name_long'] is None:
                crop_specs['name_long'] = base_name[
                        base_name.find('-', base_name.rfind('_')):]
            if crop_specs['ext'] is None:
                crop_specs['ext'] = os.path.splitext(crop_specs['fname'])[1]

        for col_name in row.index:
            if col_name not in self.io.defaults.spat_crop_cols.keys():
                crop_specs[col_name] = row[col_name]
        if not pd.notnull(crop_specs['name_long']):
            crop_specs['name_long'] = None
        if not pd.notnull(crop_specs['plot_id_ref']):
            crop_specs['plot_id_ref'] = None
        if not pd.notnull(crop_specs['name_short']):
            crop_specs['name_short'] = None

        self.crop_specs = crop_specs
        return crop_specs

    def _pix_to_mapunit(self, crop_specs, spyfile=None):
        '''
        Looks over specifications of ``crop_specs``, and converts betweeen pixel
        units and map units if one is populated and the other is ``None``
        '''
        cs = crop_specs.copy()

        if spyfile is None:
            spyfile = self.io.spyfile
        spy_ps_e = float(spyfile.metadata['map info'][5])
        spy_ps_n = float(spyfile.metadata['map info'][6])
        # Crop size
#        if cs['crop_e_pix'] is None and cs['crop_e_m'] is not None:
        if pd.isnull(cs['crop_e_pix']) and pd.notnull(cs['crop_e_m']):
            cs['crop_e_pix'] = int(cs['crop_e_m'] / spy_ps_e)
        elif pd.notnull(cs['crop_e_pix']) and pd.isnull(cs['crop_e_m']):
            cs['crop_e_m'] = cs['crop_e_pix'] * spy_ps_e
        if pd.isnull(cs['crop_n_pix']) and pd.notnull(cs['crop_n_m']):
            cs['crop_n_pix'] = int(cs['crop_n_m'] / spy_ps_n)
        elif pd.notnull(cs['crop_n_pix']) and pd.isnull(cs['crop_n_m']):
            cs['crop_n_m'] = cs['crop_n_pix'] * spy_ps_n
        # Buffer
        if pd.isnull(cs['buf_e_pix']) and pd.notnull(cs['buf_e_m']):
            cs['buf_e_pix'] = int(cs['buf_e_m'] / spy_ps_e)
        elif pd.notnull(cs['buf_e_pix']) and pd.isnull(cs['buf_e_m']):
            cs['buf_e_m'] = cs['buf_e_pix'] * spy_ps_e
        if pd.isnull(cs['buf_n_pix']) and pd.notnull(cs['buf_n_m']):
            cs['buf_n_pix'] = int(cs['buf_n_m'] / spy_ps_e)
        elif pd.notnull(cs['buf_n_pix']) and pd.isnull(cs['buf_n_m']):
            cs['buf_n_m'] = cs['buf_n_pix'] * spy_ps_e
        # Shift
        if pd.isnull(cs['gdf_shft_e_pix']) and pd.notnull(cs['gdf_shft_e_m']):
            cs['gdf_shft_e_pix'] = int(cs['gdf_shft_e_m'] / spy_ps_e)
        elif pd.notnull(cs['gdf_shft_e_pix']) and pd.isnull(cs['gdf_shft_e_m']):
            cs['gdf_shft_e_m'] = cs['gdf_shft_e_pix'] * spy_ps_e
        if pd.isnull(cs['gdf_shft_n_pix']) and pd.notnull(cs['gdf_shft_n_m']):
            cs['gdf_shft_n_pix'] = int(cs['gdf_shft_n_m'] / spy_ps_e)
        elif pd.notnull(cs['gdf_shft_n_pix']) and pd.isnull(cs['gdf_shft_n_m']):
            cs['gdf_shft_n_m'] = cs['gdf_shft_n_pix'] * spy_ps_e
        # Alley size
        if (pd.isnull(cs['alley_size_e_pix']) and
                pd.notnull(cs['alley_size_e_m'])):
            cs['alley_size_e_pix'] = int(cs['alley_size_e_m'] / spy_ps_e)
        elif (pd.notnull(cs['alley_size_e_pix']) and
                  pd.isnull(cs['alley_size_e_m'])):
            cs['alley_size_e_m'] = cs['alley_size_e_pix'] * spy_ps_e
        if (pd.isnull(cs['alley_size_n_pix']) and
                pd.notnull(cs['alley_size_n_m'])):
            cs['alley_size_n_pix'] = int(cs['alley_size_n_m'] / spy_ps_n)
        elif (pd.notnull(cs['alley_size_n_pix']) and
                  pd.isnull(cs['alley_size_n_m'])):
            cs['alley_size_n_m'] = cs['alley_size_n_pix'] * spy_ps_n
        self.crop_specs = cs
        return cs

    def _composite_band_setup(self, base_dir_out, fname, folder_name, name_append):
         '''
         '''
         if base_dir_out is None:
             base_dir = os.path.dirname(fname)
             dir_out, name_append = self._save_file_setup(
                     base_dir, folder_name, name_append)
         else:
             dir_out, name_append = self._save_file_setup(
                     base_dir_out, folder_name, name_append)
         name_print = self._get_name_print()
         return dir_out, name_print, name_append

    def _band_math_setup(self, base_dir_out, folder_name, fname, name_append,
                         method):
        '''
        '''
        msg = ('``method`` must be one of either "ndi", "ratio", "derivative", '
               'or "mcari2".\n')
        assert method in ['ndi', 'ratio', 'derivative', 'mcari2'], msg

        if base_dir_out is None:
            base_dir = os.path.dirname(fname)
            dir_out, name_append = self._save_file_setup(
                    base_dir, folder_name, name_append)
        else:
            dir_out, name_append = self._save_file_setup(
                    base_dir_out, folder_name, name_append)
        name_print = self._get_name_print()
        if method == 'ndi':
            print('\nCalculating normalized difference index for: {0}'
                  ''.format(name_print))
        elif method == 'ratio':
            print('\nCalculating simple ratio index for: {0}'
                  ''.format(name_print))
        return dir_out, name_print, name_append

    def _mask_stats_setup(self, mask_thresh, mask_percentile, mask_side):
        '''
        Parse thesholds and percentiles to dynamically set column names for
        masked df_stats
        '''
        if mask_thresh is not None:
            if not isinstance(mask_thresh, list):
                mask_thresh = [mask_thresh]
            mask_thresh_print = '-'.join([str(x) for x in mask_thresh])
        if mask_percentile is not None:
            if not isinstance(mask_percentile, list):
                mask_percentile = [mask_percentile]
            mask_pctl_print = '-'.join([str(x) for x in mask_percentile])
        if mask_side is not None:
            if not isinstance(mask_side, list):
                mask_side = [mask_side]
            mask_side_print = '-'.join([str(x) for x in mask_side])
        if mask_thresh is not None and mask_percentile is not None:
            type_mask = ('mask-{0}-thresh-{1}-pctl-{2}'.format(
                    mask_side_print, mask_thresh_print, mask_pctl_print))
        elif mask_thresh is not None and mask_percentile is None:
            type_mask = ('mask-{0}-thresh-{1}'.format(
                    mask_side_print, mask_thresh_print))
        elif mask_thresh is None and mask_percentile is not None:
            type_mask = ('mask-{0}-pctl-{1}'.format(
                    mask_side_print, mask_pctl_print))
        columns = ['fname', 'plot_id', type_mask + '-count',
                   type_mask + '-mean', type_mask + '-stdev',
                   type_mask + '-median']
        df_stats = pd.DataFrame(columns=columns)
        return df_stats, type_mask

    def _mask_single_stats(self, fname, array_bm, metadata_bm,
                           mask_thresh, mask_percentile, mask_side, df_stats):
        '''
        Creates the bandmath mask and summarizes the band math values after
        masking unwanted pixels. Returns the single masked bandmath array and
        the stats dataframe with the new image data appended as a row
        '''
        array_mask, metadata_bm = self.io.tools.mask_array(
                array_bm, metadata_bm, thresh=mask_thresh,
                percentile=mask_percentile, side=mask_side)
        # array_mask, metadata_bm = hsbatch.io.tools.mask_array(
        #         array_bandmath1, metadata_bandmath1, thresh=mask_thresh,
        #         percentile=mask_percentile, side=mask_side)

        # stat_mask_count = np.count_nonzero(~np.isnan(array_mask))
        # all nan values should be masked from mask_array() function
        stat_mask_count = array_mask.count()
        stat_mask_mean = array_mask.mean()
        stat_mask_std = array_mask.std()
        stat_mask_med = np.ma.median(array_mask)
        # stat_mask_mean = np.nanmean(array_mask)
        # stat_mask_std = np.nanstd(array_mask)
        # stat_mask_med = np.nanmedian(array_mask)

        data = [fname, self.io.name_plot, stat_mask_count, stat_mask_mean,
                stat_mask_std, stat_mask_med]
        df_stats_temp = pd.DataFrame(data=[data], columns=df_stats.columns)
        df_stats = df_stats.append(df_stats_temp, ignore_index=True)
        return array_mask, df_stats


    def _mask_two_step(self, mask_dir, mask_side, mask_thresh,
                       mask_percentile, fname, df_stats1, df_stats2,
                       name_label):
        '''
        Performs a two-step masking process. The masked masked
        bandmath arrays and stats for each step are returned.
        '''
        msg1 = ('Either ``mask_thresh`` or ``mask_percentile`` is a '
                'list, but ``mask_dir`` is not a list. If trying to '
                'perform a "two-step" masking process, please be sure '
                'to pass a list with length of two for both '
                '``mask_dir`` and ``mask_side``, as well as either '
                'for ``mask_thresh`` or ``mask_percentile``.\n'
                '``mask_dir``: {0}\n``mask_side``: {1}'
                ''.format(mask_dir, mask_side))
        msg2 = ('Either ``mask_thresh`` or ``mask_percentile`` is a '
                'list, but ``mask_side`` is not a list. If trying to '
                'perform a "two-step" masking process, please be sure '
                'to pass a list with length of two for both '
                '``mask_dir`` and ``mask_side``, as well as either '
                'for ``mask_thresh`` or ``mask_percentile``.\n'
                '``mask_dir``: {0}\n``mask_side``: {1}'
                ''.format(mask_dir, mask_side))
        assert isinstance(mask_dir, list), msg1
        assert isinstance(mask_side, list), msg2

        array_bandmath1, metadata_bandmath1 = self._get_array_similar(
                mask_dir[0])
        array_bandmath2, metadata_bandmath2 = self._get_array_similar(
                mask_dir[1])
        if isinstance(mask_thresh, list):
            array_mask1, df_stats1 = self._mask_single_stats(
                fname, array_bandmath1, metadata_bandmath1,
                mask_thresh[0], None, mask_side[0], df_stats1)
            array_mask2, df_stats2 = self._mask_single_stats(
                fname, array_bandmath2, metadata_bandmath2,
                mask_thresh[1], None, mask_side[1], df_stats2)
        elif isinstance(mask_percentile, list):
            array_mask1, df_stats1 = self._mask_single_stats(
                fname, array_bandmath1, metadata_bandmath1, None,
                mask_percentile[0], mask_side[0], df_stats1)
            array_mask2, df_stats2 = self._mask_single_stats(
                fname, array_bandmath2, metadata_bandmath2, None,
                mask_percentile[1], mask_side[1], df_stats2)
        return array_mask1, array_mask2, df_stats1, df_stats2

    def _execute_mask(self, fname_list, mask_dir, base_dir_out, folder_name,
                      name_append, write_datacube, write_spec, write_geotiff,
                      mask_thresh, mask_percentile, mask_side):
        '''
        Actually creates the mask to keep the main function a bit cleaner
        '''
        if mask_side == 'outside':  # thresh/pctl will be a list, so take care of this first
            df_stats1, type_mask1 = self._mask_stats_setup(
                mask_thresh, mask_percentile, mask_side)
            df_stats2 = None
            type_mask2 = None
        # if mask_side is not "outside" and thresh is list, then it's a 2-step
        elif isinstance(mask_thresh, list):
            if not isinstance(mask_side, list):
                maskside = [mask_side, mask_side]  # ensure that mask_side is two parts as well
            df_stats1, type_mask1 = self._mask_stats_setup(mask_thresh[0], None, mask_side[0])
            df_stats2, type_mask2 = self._mask_stats_setup(mask_thresh[1], None, mask_side[1])
        elif isinstance(mask_percentile, list):
            if not isinstance(mask_side, list):
                maskside = [mask_side, mask_side]  # ensure that mask_side is two parts as well
            df_stats1, type_mask1 = self._mask_stats_setup(None, mask_percentile[0], mask_side[0])
            df_stats2, type_mask2 = self._mask_stats_setup(None, mask_percentile[1], mask_side[1])
        else:
            df_stats1, type_mask1 = self._mask_stats_setup(mask_thresh, mask_percentile, mask_side)
            df_stats2 = None
            type_mask2 = None

        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'.format(idx, len(fname_list)))
            self.io.read_cube(fname)
            metadata = self.io.spyfile.metadata.copy()
            metadata_geotiff = self.io.spyfile.metadata.copy()
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            if self._file_exists_check(
                    dir_out, name_label, write_datacube=write_datacube,
                    write_spec=write_spec, write_geotiff=write_geotiff) is True:
                continue
            # array = self.io.spyfile.load()
            array = self.io.spyfile.open_memmap()

            if mask_dir is None:
                mask_dir = os.path.join(self.io.base_dir, 'band_math')
            if df_stats2 is not None:
                array_mask1, array_mask2, df_stats1, df_stats2 =\
                    self._mask_two_step(mask_dir, mask_side, mask_thresh,
                                        mask_percentile, fname, df_stats1,
                                        df_stats2, name_label)
                array_mask = np.logical_or(array_mask1.mask,
                                           array_mask2.mask)
            else:  # things are much simpler
                array_bandmath1, metadata_bandmath1 = self._get_array_similar(
                        mask_dir)
                array_mask, df_stats1 = self._mask_single_stats(
                    fname, array_bandmath1, metadata_bandmath1, mask_thresh,
                    mask_percentile, mask_side, df_stats1)
                array_mask = array_mask.mask

            spec_mean, spec_std, datacube_masked = self.io.tools.mean_datacube(
                    array, array_mask)
            self.spec_mean = spec_mean
            self.spec_std = spec_std
            hist_str = (" -> hs_process.batch.segment_create_mask[<"
                        "label: 'mask_thresh?' value:{0}; "
                        "label: 'mask_percentile?' value:{1}; "
                        "label: 'mask_side?' value:{2}>]"
                        "".format(mask_thresh, mask_percentile, mask_side))
            metadata['history'] += hist_str
            metadata_geotiff['history'] += hist_str

            if write_datacube is True:
                self._write_datacube(dir_out, name_label, datacube_masked,
                                     metadata)
            if write_spec is True:
                name_label_spec = (os.path.splitext(name_label)[0] +
                                   '-mean.spec')
                self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
                                 metadata)

            self.array_mask = array_mask
            if write_geotiff is True:
                self._write_geotiff(array_mask, fname, dir_out, name_label,
                                    metadata_geotiff, self.io.tools)
#        n = 1  # because multiple mask events may take place in same folder..
#        fname_csv = 'mask-stats.csv'.format(str(n).zfill(3))
#        fname_csv_full = os.path.join(dir_out, fname_csv)
#
#        while os.path.isfile(fname_csv_full):
#            n += 1
#            fname_csv = 'mask-stats-{0}.csv'.format(str(n).zfill(3))
#            dir_name, base_name = os.path.split(fname_csv_full)
##            base_name, ext = os.path.split(base_name)
#            fname_csv_full = os.path.join(dir_name, fname_csv)

        # fname_csv1 = 'mask-stats1.csv'
        if len(df_stats1) > 0:
            fname_stats1 = os.path.join(dir_out, type_mask1 + '.csv')
            df_stats1.to_csv(fname_stats1, index=False)
        if df_stats2 is not None:
            if len(df_stats2) > 0:
                # fname_csv2 = 'mask-stats2.csv'
                fname_stats2 = os.path.join(dir_out, type_mask2 + '.csv')
                df_stats2.to_csv(fname_stats2, index=False)

#
#############################################
#
#            name_label_bm = (name_print + name_append + '-{0}-{1}-{2}.'
#                             ''.format(method, int(np.mean(wl1)),
#                                       int(np.mean(wl2))) +
#                             self.io.defaults.envi_write.interleave)
#            meta_bm['label'] = name_label_bm
#
#            if mask_thresh is not None or mask_percentile is not None:
#                array_bm, meta_bm = self.my_segment.tools.mask_array(
#                        array_bm, metadata, thresh=mask_thresh,
#                        percentile=mask_percentile, side=mask_side)
#                name_lab_dc = (name_print + '-{0}-mask-{1}-{2}.'
#                               ''.format(method, int(np.mean(wl1)),
#                                         int(np.mean(wl2))) +
#                               self.io.defaults.envi_write.interleave)
#            # should we make an option to save a mean spectra as well?
#            # Yes - we aren't required to save intermediate results and do
#            # another batch process..? we get everything done in one shot -
#            # after all, why do we want to do band math if we aren't also
#            # calculating the average of the area (unless cropping hasn't
#            # been perfomed yet)?
#            # No - Keep it simpler and keep batch functions more specific in
#            # their capabilities (e.g., batch.band_math, batch.mask_array,
#            # batch.veg_spectra)
#
#            if np.ma.is_masked(array_bm):
#                # don't pass thresh, etc. because array is already masked
#                # pass the spyfile for the metadata (tainted from threshold)
#                self.io.read_cube(fname)  # read again to get fresh metadata
#                self.io.spyfile.metadata['history'] = meta_bm['history']
#                spec_mean, spec_std, datacube_masked, datacube_md =\
#                    self.my_segment.veg_spectra(
#                            array_bm, spyfile=self.io.spyfile)
#                if save_datacube is True:
#                    hdr_file = os.path.join(dir_out, name_lab_dc + '.hdr')
#                    self.io.write_cube(hdr_file, datacube_masked,
#                                       dtype=self.io.defaults.envi_write.dtype,
#                                       force=self.io.defaults.envi_write.force,
#                                       ext=self.io.defaults.envi_write.ext,
#                                       interleave=self.io.defaults.envi_write.interleave,
#                                       byteorder=self.io.defaults.envi_write.byteorder,
#                                       metadata=datacube_md)
#                if save_spec is True:
#                    spec_md = datacube_md.copy()
#                    name_label_spec = (os.path.splitext(name_lab_dc)[0] +
#                                       '-spec-mean.spec')
#                    spec_md['label'] = name_label_spec
#                    hdr_file = os.path.join(dir_out, name_label_spec + '.hdr')
#                    self.io.write_spec(hdr_file, spec_mean, spec_std,
#                                       dtype=self.io.defaults.envi_write.dtype,
#                                       force=self.io.defaults.envi_write.force,
#                                       ext=self.io.defaults.envi_write.ext,
#                                       interleave=self.io.defaults.envi_write.interleave,
#                                       byteorder=self.io.defaults.envi_write.byteorder,
#                                       metadata=spec_md)
#            self._write_datacube(dir_out, name_label_bm, array_bm, metadata)
#            if geotiff is True:
#                self._write_geotiff(array_bm, fname, dir_out, name_label_bm,
#                                    meta_bm, self.my_segment.tools)
    def _write_stats(self, dir_out, df_stats, fname_csv='stats.csv'):
        '''
        Writes df_stats to <dir_out>, ensuring lock is in place if it exists to
        work as expected with parallel processing.
        '''
        fname_stats = os.path.join(dir_out, fname_csv)

        if self.lock is not None:
            with self.lock:
                if os.path.isfile(fname_stats):
                    df_stats_in = pd.read_csv(fname_stats)
                    df_stats = df_stats_in.append(df_stats)
                df_stats.to_csv(fname_stats, index=False)
        else:
            if os.path.isfile(fname_stats):
                df_stats_in = pd.read_csv(fname_stats)
                df_stats = df_stats_in.append(df_stats)
            df_stats.to_csv(fname_stats, index=False)

    def _execute_composite_band(self, fname_list, base_dir_out, folder_name,
                                name_append, write_geotiff, wl1, b1,
                                list_range, plot_out):
        '''
        Actually executes the composit band to keep the main function a bit
        cleaner
        '''
        type_bm = '-comp-{0}'.format(int(np.mean(wl1)))
        columns = ['fname', 'plot_id', 'count', 'mean', 'std_dev', 'median',
                   'pctl_10th', 'pctl_25th', 'pctl_50th', 'pctl_75th',
                   'pctl_90th', 'pctl_95th']
        df_stats = pd.DataFrame(columns=columns)

        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'.format(idx, len(fname_list)))
            self.io.read_cube(fname)
            dir_out, name_print, name_append = self._composite_band_setup(
                    base_dir_out, fname, folder_name, name_append)
            self.my_segment = segment(self.io.spyfile)
            name_label = (name_print + name_append + type_bm + '.{0}'
                          ''.format(self.io.defaults.envi_write.interleave))
            if self._file_exists_check(
                    dir_out, name_label, write_datacube=True,
                    write_geotiff=write_geotiff, write_plot=plot_out) is True:
                continue
            array_b1, metadata = self.my_segment.composite_band(
                wl1=wl1, b1=b1, list_range=list_range, print_out=False)

            stat_count = np.count_nonzero(~np.isnan(array_b1))
            stat_mean = np.nanmean(array_b1)
            stat_std = np.nanstd(array_b1)
            stat_med = np.nanmedian(array_b1)
            stat_pctls = np.nanpercentile(array_b1, [10, 25, 50, 75, 90, 95])

            data = [fname, self.io.name_plot, stat_count, stat_mean, stat_std,
                    stat_med, stat_pctls[0], stat_pctls[1], stat_pctls[2],
                    stat_pctls[3], stat_pctls[4], stat_pctls[5]]
            df_stats_temp = pd.DataFrame(data=[data], columns=columns)
            df_stats = df_stats.append(df_stats_temp, ignore_index=True)

            if plot_out is True:
                fname_fig = os.path.join(dir_out,
                                         os.path.splitext(name_label)[0] +
                                         '.png')
                self.io.tools.plot_histogram(
                        array_b1, fname_fig=fname_fig, title=name_print,
                        xlabel=array_b1.upper(), percentile=90, bins=50,
                        fontsize=14, color='#444444')
            metadata['label'] = name_label

            self._write_datacube(dir_out, name_label, array_b1, metadata)
            if write_geotiff is True:
                self._write_geotiff(array_b1, fname, dir_out, name_label,
                                    metadata, self.my_segment.tools)
        if len(df_stats) > 0:
            self._write_stats(dir_out, df_stats, fname_csv=name_append[1:] + '-stats.csv')

    def _execute_band_math(self, fname_list, base_dir_out, folder_name,
                           name_append, write_geotiff, method, wl1, wl2, wl3, b1, b2,
                           b3, list_range, plot_out):
        '''
        Actually executes the band math to keep the main function a bit
        cleaner
        '''
        if method == 'ndi' or method == 'ratio':
            type_bm = ('{0}-{1}-{2}'.format(method, int(np.mean(wl1)),
                                            int(np.mean(wl2))))
        elif method == 'derivative':
            type_bm = ('{0}-{1}-{2}-{3}'.format(method, int(np.mean(wl1)),
                                                int(np.mean(wl2)),
                                                int(np.mean(wl2))))
        elif method == 'mcari2':
            type_bm = ('{0}-{1}-{2}-{3}'.format(method, int(np.mean(wl1)),
                                                int(np.mean(wl2)),
                                                int(np.mean(wl2))))
        columns = ['fname', 'plot_id', 'count', 'mean', 'std_dev', 'median',
                   'pctl_10th', 'pctl_25th', 'pctl_50th', 'pctl_75th',
                   'pctl_90th', 'pctl_95th']
        df_stats = pd.DataFrame(columns=columns)

        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'.format(idx, len(fname_list)))
            self.io.read_cube(fname)
            dir_out, name_print, name_append = self._band_math_setup(
                    base_dir_out, folder_name, fname, name_append, method)
            self.my_segment = segment(self.io.spyfile)

            if method == 'ndi':
                name_label = (name_print + name_append + '-{0}-{1}-{2}.{3}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        self.io.defaults.envi_write.interleave))
                if self._file_exists_check(
                        dir_out, name_label, write_datacube=True,
                        write_geotiff=write_geotiff,
                        write_plot=plot_out) is True:
                    continue
                array_bm, metadata = self.my_segment.band_math_ndi(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=False)
            elif method == 'ratio':
                name_label = (name_print + name_append + '-{0}-{1}-{2}.{3}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        self.io.defaults.envi_write.interleave))
                if self._file_exists_check(
                        dir_out, name_label, write_datacube=True,
                        write_geotiff=write_geotiff,
                        write_plot=plot_out) is True:
                    continue
                array_bm, metadata = self.my_segment.band_math_ratio(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=False)
            elif method == 'derivative':
                name_label = (name_print + name_append + '-{0}-{1}-{2}-{3}.{4}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        int(np.mean(wl3)),
                                        self.io.defaults.envi_write.interleave))
                if self._file_exists_check(
                        dir_out, name_label, write_datacube=True,
                        write_geotiff=write_geotiff,
                        write_plot=plot_out) is True:
                    continue
                array_bm, metadata = self.my_segment.band_math_derivative(
                        wl1=wl1, wl2=wl2, wl3=wl3, b1=b1, b2=b2, b3=b3,
                        list_range=list_range, print_out=False)
            elif method == 'mcari2':
                name_label = (name_print + name_append + '-{0}-{1}-{2}-{3}.{4}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        int(np.mean(wl3)),
                                        self.io.defaults.envi_write.interleave))
                if self._file_exists_check(
                        dir_out, name_label, write_datacube=True,
                        write_geotiff=write_geotiff,
                        write_plot=plot_out) is True:
                    continue
                array_bm, metadata = self.my_segment.band_math_mcari2(
                        wl1=wl1, wl2=wl2, wl3=wl3, b1=b1, b2=b2, b3=b3,
                        list_range=list_range, print_out=False)

            stat_count = np.count_nonzero(~np.isnan(array_bm))
            stat_mean = np.nanmean(array_bm)
            stat_std = np.nanstd(array_bm)
            stat_med = np.nanmedian(array_bm)
            stat_pctls = np.nanpercentile(array_bm, [10, 25, 50, 75, 90, 95])

            data = [fname, self.io.name_plot, stat_count, stat_mean, stat_std,
                    stat_med, stat_pctls[0], stat_pctls[1], stat_pctls[2],
                    stat_pctls[3], stat_pctls[4], stat_pctls[5]]
            df_stats_temp = pd.DataFrame(data=[data], columns=columns)
            df_stats = df_stats.append(df_stats_temp, ignore_index=True)

            if plot_out is True:
                fname_fig = os.path.join(dir_out,
                                         os.path.splitext(name_label)[0] +
                                         '.png')
                self.io.tools.plot_histogram(
                        array_bm, fname_fig=fname_fig, title=name_print,
                        xlabel=type_bm.upper(), percentile=90, bins=50,
                        fontsize=14, color='#444444')
#                self._plot_histogram(array_bm, fname_fig, title=name_print,
#                                     xlabel=type_bm.upper(), percentile=90,
#                                     fontsize=14,
#                                     color='#444444')

            metadata['label'] = name_label

            self._write_datacube(dir_out, name_label, array_bm, metadata)
            if write_geotiff is True:
                self._write_geotiff(array_bm, fname, dir_out, name_label,
                                    metadata, self.my_segment.tools)

        if len(df_stats) > 0:
            self._write_stats(dir_out, df_stats, fname_csv=name_append[1:] + '-stats.csv')


    def _get_ndvi_simple(self, df_class_spec, n_classes, plot_out=True):
        '''
        Find kmeans class with lowest NDVI, which represents the soil class
        '''
        nir_b = self.io.tools.get_band(760)
        re_b = self.io.tools.get_band(715)
        red_b = self.io.tools.get_band(681)
        green_b = self.io.tools.get_band(555)

        nir = df_class_spec.iloc[nir_b]
        re = df_class_spec.iloc[re_b]
        red = df_class_spec.iloc[red_b]
        green = df_class_spec.iloc[green_b]

        df_ndvi = (nir-red)/(nir+red)
        class_soil = df_ndvi[df_ndvi == df_ndvi.min()].index[0]
        class_veg = df_ndvi[df_ndvi == df_ndvi.max()].index[0]
        if plot_out is True:
            df_class_spec['wavelength'] = self.io.tools.meta_bands.values()
            fig, ax = plt.subplots()
            sns.lineplot(data=df_class_spec, ax=ax)
            legend = ax.legend()
            legend.set_title('K-means classes')
            legend.texts[class_soil].set_text('Soil')
            legend.texts[class_veg].set_text('Vegetation')
        return class_soil, class_veg

    def _execute_kmeans(self, fname_list, base_dir_out, folder_name,
                        name_append, geotiff, n_classes, max_iter,
                        plot_out):
        '''
        Actually executes the kmeans clustering to keep the main function a bit
        cleaner
        '''
        columns = ['fname']
        for i in range(n_classes):
            columns.append('class_{0}_count'.format(i))
        for i in range(n_classes):
            columns.append('class_{0}_ndvi'.format(i))
        for i in range(n_classes):
            columns.append('class_{0}_gndvi'.format(i))
        for i in range(n_classes):
            columns.append('class_{0}_ndre'.format(i))
        for i in range(n_classes):
            columns.append('class_{0}_mcari2'.format(i))
        df_stats = pd.DataFrame(columns=columns)
        for fname in fname_list:
            self.io.read_cube(fname)
            metadata = self.io.spyfile.metadata
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            self.my_segment = segment(self.io.spyfile)

            array_class, df_class_spec, metadata = self.my_segment.kmeans(
                    n_classes=n_classes, max_iter=max_iter,
                    spyfile=self.io.spyfile)

            data = [fname]
            nir_b = self.my_segment.tools.get_band(800)
            re_b = self.my_segment.tools.get_band(720)
            red_b = self.my_segment.tools.get_band(670)
            green_b = self.my_segment.tools.get_band(550)
            nir = df_class_spec.iloc[nir_b]
            re = df_class_spec.iloc[re_b]
            red = df_class_spec.iloc[red_b]
            green = df_class_spec.iloc[green_b]
            df_ndvi = (nir - red) / (nir + red)
            df_gndvi = (nir - green) / (nir + green)
            df_ndre = (nir - re) / (nir + re)
            df_mcari2 = ((1.5 * (2.5 * (nir - red) - 1.3 * (nir - green))) /
                np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * np.sqrt(red)) - 0.5))
            for i in range(n_classes):
                class_n = len(array_class[array_class == i])
                pix_n = len(array_class.flatten())
                if class_n / pix_n < 0.05:  # if < 5% of pixels in a class
                    df_ndvi[i] = np.nan
                    df_gndvi[i] = np.nan
                    df_ndre[i] = np.nan
                    df_mcari2[i] = np.nan
                data.append(len(array_class[array_class == i]))
            for ndvi in df_ndvi:
                data.append(ndvi)
            for gndvi in df_gndvi:
                data.append(gndvi)
            for ndre in df_ndre:
                data.append(ndre)
            for mcari2 in df_mcari2:
                data.append(mcari2)

            df_stats_temp = pd.DataFrame(data=[data], columns=columns)
            df_stats = df_stats.append(df_stats_temp, ignore_index=True)

            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)

            if plot_out is True:
                df_class_spec['wavelength'] = self.io.tools.meta_bands.values()
                df_class_spec.set_index('wavelength', drop=True, inplace=True)
                fig, ax = plt.subplots()
                sns.lineplot(data=df_class_spec*100, ax=ax)
                ax.set_title(os.path.basename(name_label))
                ax.set_xlabel('Wavelength')
                ax.set_ylabel('Reflectance (%)')
                legend = ax.legend()
                legend.set_title('K-means classes')
#                legend.texts[class_soil].set_text('Soil')
#                legend.texts[class_veg].set_text('Vegetation')
                fname_fig = os.path.join(dir_out, name_print + name_append +
                                         '.png')
                fig.savefig(fname_fig)

            self._write_datacube(dir_out, name_label, array_class, metadata)

#            name_label_spec = (os.path.splitext(name_label)[0] +
#                                       '-spec-mean.spec')
#            self._write_spec(dir_out, name_label, spec_mean, spec_std,
#                            metadata)
            if geotiff is True:
                self._write_geotiff(array_class, fname, dir_out, name_label,
                                    metadata, self.my_segment.tools)

#            if mask_soil is True:
#                class_soil, class_veg = self._get_ndvi_simple(
#                        df_class_spec, n_classes, plot_out=True)
#                array_class = np.ma.masked_where(array_class==class_soil,
#                                                 array_class)
#                name_label = (name_print + name_append + '-mask-soil' + '.' +
#                              self.io.defaults.envi_write.interleave)
#                self._write_datacube(dir_out, name_label, array_class,
#                                     metadata)

        fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
        if os.path.isfile(fname_stats) and self.io.defaults.envi_write.force is False:
            df_stats_in = pd.read_csv(fname_stats)
            df_stats = df_stats_in.append(df_stats)
        df_stats.to_csv(fname_stats, index=False)

    def _crop_check_input(self, fname_sheet, fname_list, method):
        '''
        Checks that either `fname_sheet` or `fname_list` were passed (and not
        both)
        '''
        if fname_sheet is not None:
            if isinstance(fname_sheet, pd.DataFrame) and pd.isnull(fname_list):
                df_plots = fname_sheet
            elif os.path.splitext(fname_sheet)[-1] == '.csv' and pd.isnull(fname_list):
                df_plots = pd.read_csv(fname_sheet)
            elif fname_list is not None:
                msg2 = ('Both ``fname_sheet`` and ``fname_list`` were passed. '
                        '``fname_list`` (perhaps from ``base_dir``) will be '
                        'ignored.\n')
                print(msg2)
                if isinstance(fname_sheet, pd.DataFrame):
                    df_plots = fname_sheet
                elif os.path.splitext(fname_sheet)[-1] == '.csv':
                    df_plots = pd.read_csv(fname_sheet)
            return df_plots
        elif pd.isnull(fname_sheet) and pd.isnull(fname_list):
            msg1 = ('Neither ``fname_sheet`` nor ``fname_list`` were passed. '
                    'Please pass one or the other (not both) and run '
                    '``batch.spatial_crop`` again.\n')
            raise TypeError(msg1)
        else:  # fname_list was passed and df_plots will be figured out later
            msg3 = ('``method`` is "single", but ``fname_list`` was passed '
                    'instead of ``fname_sheet``.\n\nIf performing '
                    '``crop_single``, please pass ``fname_sheet``.\n\nIf '
                    'performing ``crop_many_gdf``, please pass ``fname_list`` '
                    '(perhaps via ``base_dir``).\n')
            assert method in ['many_grid', 'many_gdf'], msg3
            return

    def _file_exists_check(self, dir_out, name_label,
                           write_datacube=False, write_spec=False,
                           write_geotiff=False, write_plot=False):
        '''
        Checks if all files to be created exist already; if so, returns True;
        if not, returns False.
        '''
        if self.io.defaults.envi_write.force is True:
            return False

        write_dict = {'write_datacube': write_datacube,
                      'write_spec': write_spec,
                      'write_geotiff': write_geotiff,
                      'write_plot': write_plot}
        ext_dict = {'write_datacube': '.bip',
                    'write_spec': '-mean.spec',
                    'write_geotiff': '.tif',
                    'write_plot': '.png'}

        msg = ('Skipping file - it appears as if this image has already '
               'been processed. Overwrite files by passing out_force=True\n'
               'Filename (short): {0}'.format(name_label))

        for key in write_dict.keys():
            if write_dict[key] is True:
                fname = os.path.splitext(
                    os.path.join(dir_out, name_label))[0] + ext_dict[key]
                # if key == 'write_spec':
                # fname = os.path.splitext(fname)[0] + b[key])
                if not os.path.isfile(fname):  # if we need the file and it doesn't exist, we're done checking
                    return False
        # if we get here without already exiting, all files should exist
        print(msg)
        return True

        # else:
        #     return False
        # if write_datacube is True:
        #     data_file = os.path.join(dir_out, name_label)
        #     hdr_file = os.path.join(dir_out, name_label + '.hdr')
        # if write_spec is True:
        #     spec_file = os.path.join(
        #         dir_out, os.path.splitext(name_label)[0] + '-mean.spec')
        # if write_geotiff is True:
        #     tif_file = os.path.join(
        #         dir_out, os.path.splitext(name_label)[0] + '.tif')
        # if write_plot is True:
        #     png_file = os.path.join(
        #         dir_out, os.path.splitext(name_label)[0] + '.png')
        # if write_geotiff is True:

        # msg = ('Skipping file - it appears as if this image has already '
        #        'been processed. Overwrite files by passing out_force=True to '
        #        'the ``batch`` function.\nFunction: {0}'
        #        ''.format(function))

        # if function is None:  # do a generic check for the datacube
        #     if os.path.isfile(data_file):
        #         print(msg)
        #         return True
        #     else:
        #         return False
        # elif function == 'spatial_crop':
        #     if (os.path.isfile(data_file) and os.path.isfile(hdr_file) and
        #         os.path.isfile(tif_file)):
        #         print(msg)
        #         return True
        #     else:
        #         return False
        # elif function == 'segment_band_math':
        #     if (os.path.isfile(data_file) and os.path.isfile(hdr_file) and
        #         os.path.isfile(tif_file) and os.path.isfile(png_file)):
        #         print(msg)
        #         return True
        #     else:
        #         return False
        # elif function == 'segment_composite_band':

        # # if (os.path.isfile(data_file) and os.path.isfile(hdr_file) and
        # #     os.path.isfile(fname_tif) and
        # #     self.io.defaults.envi_write.force is False):
        # #     print(msg)
        # #     return True
        # else:
        #     return False

    def _crop_loop(self, df_plots, gdf, base_dir_out, folder_name,
                   name_append, write_geotiff):
        '''
        ``df_plots`` is assumed to contain all the necessary information to
        crop *each plot* from an image or from multiple images. In other words,
        _crop_loop() will perform a single cropping procedure (via
        ``spatial_mod.crop_single()``) for each row in ``df_plots``. Thus,
        all the necessary information should be contained in df_plots to run
        crop_single(). This function is not meant for dataframes containing
        information to perform crop_many(), so be sure to hone in on that
        information before passing ``_crop_loop``.
        '''
        df_iter = tqdm(df_plots.iterrows(), total=df_plots.shape[0]) if self.progress_bar is True else df_plots.iterrows()
        for idx, row in df_iter:
            if self.progress_bar is True:
                df_iter.set_description('Processing file {0}/{1}'.format(idx, len(df_plots)))
            cs = self._crop_read_sheet(row)
            fname = os.path.join(cs['directory'], cs['fname'])
            # print('\nSpatially cropping: {0}'.format(fname))
            name_long = cs['name_long']  # ``None`` if it was never set
            plot_id_ref = cs['plot_id_ref']
            name_short = cs['name_short']
            fname_hdr = fname + '.hdr'
            self.io.read_cube(fname_hdr, name_long=name_long,
                              name_plot=plot_id_ref, name_short=name_short)
            self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
            self.my_spatial_mod.defaults = self.io.defaults
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        cs['directory'], folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            name_label = self._get_name_label(row, name_print, name_append)
            if self._file_exists_check(
                    dir_out, name_label, write_datacube=True,
                    write_geotiff=write_geotiff) is True:
                continue

            cs = self._pix_to_mapunit(cs)
            self.cs = cs
#            if method == 'single':
            # print(cs)
            array_crop, metadata = self.my_spatial_mod.crop_single(
                    pix_e_ul=cs['pix_e_ul'], pix_n_ul=cs['pix_n_ul'],
                    crop_e_pix=cs['crop_e_pix'], crop_n_pix=cs['crop_n_pix'],
                    buf_e_pix=cs['buf_e_pix'], buf_n_pix=cs['buf_n_pix'],
                    gdf_shft_e_pix=cs['gdf_shft_e_pix'], gdf_shft_n_pix=cs['gdf_shft_n_pix'],
                    plot_id_ref=plot_id_ref, gdf=gdf)

            fname = os.path.join(cs['directory'], cs['fname'])
            self._write_datacube(dir_out, name_label, array_crop, metadata)
            if write_geotiff is True:
                self._write_geotiff(array_crop, fname, dir_out, name_label,
                                    metadata, self.my_spatial_mod.tools,
                                    show_img=False)

    def _append_cropping_details(self, df_plots_many, row):
        '''
        Appends all "row" columns to df_plots_many so they carry through
        '''
        # cropping_detail_list = ['crop_e_m', 'crop_e_m', 'crop_e_pix',
        #                         'crop_n_pix', 'buf_e_m', 'buf_n_m',
        #                         'buf_e_pix', 'buf_n_pix']
        for col in row.keys():
            if pd.notnull(row[col]) and col not in df_plots_many.columns:
            # if pd.notnull(row[col]) and col in cropping_detail_list:
                # print('{0}: {1}'.format(col, row[col]))
                df_plots_many[col] = row[col]
        # if 'crop_e_m' in row.keys():  # use value from row/sheet instead of gdf
        #     df_plots_many['crop_e_m'] = row['crop_e_m']
        #     df_plots_many['crop_e_pix'] = np.nan
        # if 'crop_n_m' in row.keys():  # use value from row/sheet instead of gdf
        #     df_plots_many['crop_n_m'] = row['crop_n_m']
        #     df_plots_many['crop_n_pix'] = np.nan
        # if 'crop_e_pix' in row.keys():  # use value from row/sheet instead of gdf
        #     df_plots_many['crop_e_pix'] = row['crop_e_pix']
        #     df_plots_many['crop_e_m'] = np.nan
        # if 'crop_n_pix' in row.keys():  # use value from row/sheet instead of gdf
        #     df_plots_many['crop_n_pix'] = row['crop_n_pix']
        #     df_plots_many['crop_n_m'] = np.nan
        return df_plots_many

    def _crop_many_read_row(self, row, gdf, method):
        '''
        Helper function for reading a row of a dataframe with information about
        how to crop an image many times
        '''
        cs = self._crop_read_sheet(row)  # this function creates cs['fname']
        fname_in = os.path.join(cs['directory'], cs['fname'])
        print('Filename: {0}'.format(fname_in))
        name_long = cs['name_long']  # ``None`` if it was never set
        plot_id_ref = cs['plot_id_ref']
        name_short = cs['name_short']
        fname_hdr = fname_in + '.hdr'
        self.io.read_cube(fname_hdr, name_long=name_long,
                          name_plot=plot_id_ref, name_short=name_short)
        self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
        self.my_spatial_mod.defaults = self.io.defaults
        if method == 'many_gdf':
            df_plots_many = self._many_gdf(cs)
        elif method == 'many_grid':
            df_plots_many = self._many_grid(cs)
        else:
            msg = ('``method`` must be either "many_gdf" or "many_grid".\n'
                   'Method: {0}'.format(method))
            raise ValueError(msg)
        df_plots_many = self._append_cropping_details(df_plots_many, row)
        return df_plots_many

    def _many_grid(self, cs):
        '''Wrapper to get consice access to ``spatial_mod.crop_many_grid()'''
        df_plots = self.my_spatial_mod.crop_many_grid(
            cs['plot_id_ref'], pix_e_ul=cs['pix_e_ul'], pix_n_ul=cs['pix_n_ul'],
            crop_e_m=cs['crop_e_m'], crop_n_m=cs['crop_n_m'],
            alley_size_n_m=cs['alley_size_n_m'], buf_e_m=cs['buf_e_m'],
            buf_n_m=cs['buf_n_m'], n_plots_x=cs['n_plots_x'],
            n_plots_y=cs['n_plots_y'])
        return df_plots

    def _many_gdf(self, cs):
        '''
        Wrapper to get consice access to ``spatial_mod.crop_many_gdf();
        ``my_spatial_mod`` already has access to ``spyfile`` and ``gdf``, so no
        need to pass them here.

        If the buffer settings are None, but there are default settings for
        them, they are passed here
        '''
        if cs['plot_id_ref'] is None:
            cs['plot_id_ref'] = self.io.defaults.crop_defaults.plot_id_ref
        # if cs['buf_e_m'] is None:
        #     cs['buf_e_m'] = self.io.defaults.crop_defaults.buf_e_m
        # if cs['buf_n_m'] is None:
        #     cs['buf_n_m'] = self.io.defaults.crop_defaults.buf_n_m
        # if cs['buf_e_pix'] is None:
        #     cs['buf_e_pix'] = self.io.defaults.crop_defaults.buf_e_pix
        # if cs['buf_n_pix'] is None:
        #     cs['buf_n_pix'] = self.io.defaults.crop_defaults.buf_n_pix
        # Note: batch.spatial_crop does not consider crop_defaults for crop_X
        # or buf_X

        # df_plots = self.my_spatial_mod.crop_many_gdf(
        #     plot_id_ref=cs['plot_id_ref'], pix_e_ul=cs['pix_e_ul'],
        #     pix_n_ul=cs['pix_n_ul'], n_plots=cs['n_plots'])
        df_plots = self.my_spatial_mod.crop_many_gdf(
            plot_id_ref=cs['plot_id_ref'], pix_e_ul=cs['pix_e_ul'],
            pix_n_ul=cs['pix_n_ul'], n_plots=cs['n_plots'],
            crop_e_m=cs['crop_e_m'], crop_n_m=cs['crop_n_m'],
            crop_e_pix=cs['crop_e_pix'], crop_n_pix=cs['crop_n_pix'],
            buf_e_m=cs['buf_e_m'], buf_n_m=cs['buf_n_m'],
            buf_e_pix=cs['buf_e_pix'], buf_n_pix=cs['buf_n_pix'],
            gdf_shft_e_m=cs['gdf_shft_e_m'], gdf_shft_n_m=cs['gdf_shft_n_m'],
            gdf_shft_e_pix=cs['gdf_shft_e_pix'], gdf_shft_n_pix=cs['gdf_shft_n_pix'])
        return df_plots

    def _crop_check_files(self, df_plots):
        '''
        If file already exists and out_force is False, removes that file from
        the df_plots
        '''
        df_plots.reset_index(inplace=True)
        df_plots_out = df_plots.copy()
        for idx, row in df_plots.iterrows():
            fname = os.path.join(row['directory'], row['name_short'] +
                                 row['name_long'] + row['ext'])
            if os.path.isfile(fname):
                df_plots_out.drop(idx, inplace=True)
        df_plots_out.reset_index(inplace=True)
        return df_plots_out

    def _execute_crop(self, fname_sheet, fname_list, base_dir_out, folder_name,
                      name_append, write_geotiff, method, gdf):
        '''
        Actually executes the spatial crop to keep the main function a bit
        cleaner

        Either `fname_sheet` or `fname_list` should be None
        '''
        df_plots = self._crop_check_input(fname_sheet, fname_list, method)
        if not pd.isnull(df_plots):
            if 'date' in df_plots.columns and isinstance(df_plots['date'], str):
                df_plots['date'] = pd.to_datetime(df_plots['date'])
        if method == 'single':
            # self._crop_loop(df_plots)
            self._crop_loop(df_plots, gdf, base_dir_out, folder_name,
                   name_append, write_geotiff)
        elif method == 'many_gdf' and isinstance(df_plots, pd.DataFrame):
            # if user passes a dataframe, just do whatever it says..
            # loop through each row, doing crop_many_gdf() on each row with
            # whatever parameters are passed via the columns..
            # we should assume that each row of df_plots contains an image that
            # should have crop_many_gdf performed on it to create a new
            # dataframe that can be passed to _crop_loop()
            for idx, row in df_plots.iterrows():
                print('Computing information to spatially crop via '
                      '``spatial_mod.crop_many_gdf``:')
                df_plots_many = self._crop_many_read_row(row, gdf, method)
                self.df_plots_many = df_plots_many
                self._crop_loop(df_plots_many, gdf, base_dir_out, folder_name,
                                name_append, write_geotiff)
        elif method == 'many_gdf' and df_plots is None:
            print('Because ``fname_list`` was passed instead of '
                  '``fname_sheet``, there is not a way to infer the study '
                  'name and date. Therefore, "study" and "date" will be '
                  'omitted from the output file name. If you would like '
                  'output file names to include "study" and "date", please '
                  'pass ``fname_sheet`` with "study" and "date" columns.\n')
            for fname_in in fname_list:
                self.io.read_cube(fname_in)
                self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
                self.my_spatial_mod.defaults = self.io.defaults
                df_plots_many = self.my_spatial_mod.crop_many_gdf()
                self._crop_loop(df_plots_many, gdf, base_dir_out, folder_name,
                                name_append, write_geotiff)
        elif method == 'many_grid' and isinstance(df_plots, pd.DataFrame):
            for idx, row in df_plots.iterrows():
                print('\nComputing information to spatially crop via '
                      '``spatial_mod.crop_many_grid``:')
                df_plots_many = self._crop_many_read_row(row, gdf, method)
                self._crop_loop(df_plots_many, gdf, base_dir_out, folder_name,
                                name_append, write_geotiff)
        else:
            msg = ('Either ``method`` or ``df_plots`` are not defined '
                   'correctly. If using "many_grid" method, please be sure '
                   '``df_plots`` is being populated correcty\n\n``method``: '
                   '{0}'.format(method))
            raise ValueError(msg)

    def _write_datacube(self, dir_out, name_label, array, metadata):
        '''
        Writes a datacube to file using ``hsio.write_cube()``
        '''
        metadata['label'] = name_label
        hdr_file = os.path.join(dir_out, name_label + '.hdr')
        self.io.write_cube(hdr_file, array, metadata=metadata,
                           dtype=self.io.defaults.envi_write.dtype,
                           force=self.io.defaults.envi_write.force,
                           ext=self.io.defaults.envi_write.ext,
                           interleave=self.io.defaults.envi_write.interleave,
                           byteorder=self.io.defaults.envi_write.byteorder)

    def _write_geotiff(self, array, fname, dir_out, name_label, metadata,
                       tools, show_img=False):
        metadata['label'] = name_label
        msg = ('Projection and Geotransform information are required for '
               'writing the geotiff. This comes from the input filename, '
               'so please be sure the correct filename is passed to '
               '``fname``.\n')
        assert fname is not None and os.path.isfile(fname), msg
        fname_tif = os.path.join(dir_out,
                                 os.path.splitext(name_label)[0] + '.tif')
        img_ds = self.io._read_envi_gdal(fname_in=fname)
        projection_out = img_ds.GetProjection()
#            geotransform_out = img_ds.GetGeotransform()
        img_ds = None  # I only want to use GDAL when I have to..

        map_set = metadata['map info']
        ul_x_utm = tools.get_meta_set(map_set, 3)
        ul_y_utm = tools.get_meta_set(map_set, 4)
        size_x_m = tools.get_meta_set(map_set, 5)
        size_y_m = tools.get_meta_set(map_set, 6)
        # Note the last pixel size must be negative to begin at upper left
        geotransform_out = [ul_x_utm, size_x_m, 0.0, ul_y_utm, 0.0,
                            -size_y_m]
        self.io.write_tif(fname_tif, spyfile=array,
                          projection_out=projection_out,
                          geotransform_out=geotransform_out,
                          show_img=show_img)
#            if method == 'spatial':
#                ul_x_utm = self.my_spatial_mod.tools.get_meta_set(map_set, 3)
#                ul_y_utm = self.my_spatial_mod.tools.get_meta_set(map_set, 4)
#                size_x_m = self.my_spatial_mod.tools.get_meta_set(map_set, 5)
#                size_y_m = self.my_spatial_mod.tools.get_meta_set(map_set, 6)
#            if method == 'segment':
#                ul_x_utm = self.my_segment.tools.get_meta_set(map_set, 3)
#                ul_y_utm = self.my_segment.tools.get_meta_set(map_set, 4)
#                size_x_m = self.my_segment.tools.get_meta_set(map_set, 5)
#                size_y_m = self.my_segment.tools.get_meta_set(map_set, 6)
#            if method == 'spectral':
#                ul_x_utm = self.my_spectral_mod.tools.get_meta_set(map_set, 3)
#                ul_y_utm = self.my_spectral_mod.tools.get_meta_set(map_set, 4)
#                size_x_m = self.my_spectral_mod.tools.get_meta_set(map_set, 5)
#                size_y_m = self.my_spectral_mod.tools.get_meta_set(map_set, 6)

    def _write_spec(self, dir_out, name_label, spec_mean, spec_std,
                    metadata):
        metadata['label'] = name_label
        hdr_file = os.path.join(dir_out, name_label + '.hdr')
        self.io.write_spec(hdr_file, spec_mean, spec_std,
                           dtype=self.io.defaults.envi_write.dtype,
                           force=self.io.defaults.envi_write.force,
                           ext=self.io.defaults.envi_write.ext,
                           interleave=self.io.defaults.envi_write.interleave,
                           byteorder=self.io.defaults.envi_write.byteorder,
                           metadata=metadata)

    def _execute_spec_clip(self, fname_list, base_dir_out, folder_name,
                           name_append, wl_bands):
        '''
        Actually executes the spectral clip to keep the main function a bit
        cleaner
        '''
        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'.format(idx, len(fname_list)))
            # print('\nSpectrally clipping: {0}'.format(fname))
            # options for io.read_cube():
            # name_long, name_plot, name_short, individual_plot, overwrite
            self.io.read_cube(fname)
            self.my_spectral_mod = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            array_clip, metadata = self.my_spectral_mod.spectral_clip(
                    wl_bands=wl_bands)

            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_clip,
                               dtype=self.io.defaults.envi_write.dtype,
                               force=self.io.defaults.envi_write.force,
                               ext=self.io.defaults.envi_write.ext,
                               interleave=self.io.defaults.envi_write.interleave,
                               byteorder=self.io.defaults.envi_write.byteorder,
                               metadata=metadata)

    def _execute_spec_clip_pp(self, fname, base_dir_out, folder_name, name_append, wl_bands):
        '''
        Actually executes the spectral clip to keep the main function a bit
        cleaner
        '''
        # print('Arglist: {0}'.format(arg_list))
        # fname, base_dir_out, folder_name, name_append, wl_bands = arg_list
        print('\nSpectrally clipping: {0}'.format(fname))
        # options for io.read_cube():
        # name_long, name_plot, name_short, individual_plot, overwrite
        self.io.read_cube(fname)
        self.my_spectral_mod = spec_mod(self.io.spyfile)
        base_dir = os.path.dirname(fname)
        if base_dir_out is None:
            dir_out, name_append = self._save_file_setup(
                    base_dir, folder_name, name_append)
        else:
            dir_out, name_append = self._save_file_setup(
                    base_dir_out, folder_name, name_append)
        name_print = self._get_name_print()
        array_clip, metadata = self.my_spectral_mod.spectral_clip(
                wl_bands=wl_bands)

        name_label = (name_print + name_append + '.' +
                      self.io.defaults.envi_write.interleave)
        metadata['label'] = name_label

        hdr_file = os.path.join(dir_out, name_label + '.hdr')
        self.io.write_cube(hdr_file, array_clip,
                           dtype=self.io.defaults.envi_write.dtype,
                           force=self.io.defaults.envi_write.force,
                           ext=self.io.defaults.envi_write.ext,
                           interleave=self.io.defaults.envi_write.interleave,
                           byteorder=self.io.defaults.envi_write.byteorder,
                           metadata=metadata)
        # return hdr_file

    def _execute_spec_combine(self, fname_list, base_dir_out):
        '''
        Actually executes the spectra combine to keep the main function a bit
        cleaner
        '''
        df_specs = None
        if base_dir_out is None:
            base_dir_out = os.path.dirname(fname_list[0])
        pix_n = 0
        for fname in fname_list:
            self.io.read_spec(fname)
            spy_mem = self.io.spyfile_spec.open_memmap()
            pix_n += (np.count_nonzero(~np.isnan(spy_mem)) /
                      self.io.spyfile_spec.nbands)
        print('Combining datacubes/spectra into a single mean spectra.\n'
              'Number of input datacubes/spectra: {0}\nTotal number of '
              'pixels: {1}'
              ''.format(len(fname_list), int(pix_n)))
        for fname in fname_list:
            self.io.read_spec(fname)
            array = self.io.spyfile_spec.load()

            if len(array.shape) == 3:
                pixels = array.reshape((array.shape[0]*array.shape[1]),
                                       array.shape[2])
            else:
                pixels = array.reshape((array.shape[0]), array.shape[2])
            if df_specs is None:
                df_specs = pd.DataFrame(pixels, dtype=float)
            else:
                df_temp = pd.DataFrame(pixels, dtype=float)
                df_specs = df_specs.append(df_temp, ignore_index=True)

        self.df_mean = df_specs.mean()
        self.df_mean = self.df_mean.rename('mean')
        self.df_std = df_specs.std()
        self.df_std = self.df_std.rename('std')
        df_cv = self.df_mean / self.df_std
        df_cv = df_cv.rename('cv')

        hdr_file = os.path.join(base_dir_out, 'spec_mean_spy.spec.hdr')
        self.io.write_spec(hdr_file, self.df_mean, self.df_std,
                           dtype=self.io.defaults.envi_write.dtype,
                           force=self.io.defaults.envi_write.force,
                           ext=self.io.defaults.envi_write.ext,
                           interleave=self.io.defaults.envi_write.interleave,
                           byteorder=self.io.defaults.envi_write.byteorder,
                           metadata=self.io.spyfile_spec.metadata)

    def _execute_spec_mimic(self, fname_list, base_dir_out, folder_name,
                            name_append, sensor, df_band_response, col_wl,
                            center_wl):
        '''
        Actually executes the spectral resample to keep the main function a bit
        cleaner.
        '''
        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'
                                             ''.format(idx, len(fname_list)))
            self.io.read_cube(fname)
            self.my_spectral_mod = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            array_mimic, metadata = self.my_spectral_mod.spectral_mimic(
                    sensor=sensor, df_band_response=df_band_response,
                    col_wl=col_wl, center_wl=center_wl)

            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_mimic,
                               dtype=self.io.defaults.envi_write.dtype,
                               force=self.io.defaults.envi_write.force,
                               ext=self.io.defaults.envi_write.ext,
                               interleave=self.io.defaults.envi_write.interleave,
                               byteorder=self.io.defaults.envi_write.byteorder,
                               metadata=metadata)

    def _execute_spec_resample(self, fname_list, base_dir_out, folder_name,
                               name_append, bandwidth, bins_n):
        '''
        Actually executes the spectral resample to keep the main function a bit
        cleaner.
        '''
        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'
                                             ''.format(idx, len(fname_list)))
            self.io.read_cube(fname)
            self.my_spectral_mod = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            array_bin, metadata = self.my_spectral_mod.spectral_resample(
                    bandwidth=bandwidth, bins_n=bins_n)

            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_bin,
                               dtype=self.io.defaults.envi_write.dtype,
                               force=self.io.defaults.envi_write.force,
                               ext=self.io.defaults.envi_write.ext,
                               interleave=self.io.defaults.envi_write.interleave,
                               byteorder=self.io.defaults.envi_write.byteorder,
                               metadata=metadata)

    def _execute_spec_smooth(self, fname_list, base_dir_out, folder_name,
                             name_append, window_size, order, stats):
        '''
        Actually executes the spectral smooth to keep the main function a bit
        cleaner
        '''
        if stats is True:
            df_smooth_stats = pd.DataFrame(
                    columns=['fname', 'mean', 'std', 'cv'])

        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'.format(idx, len(fname_list)))
            # print('\nSpectrally smoothing: {0}'.format(fname))
            self.io.read_cube(fname)
            self.my_spectral_mod = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            array_smooth, metadata = self.my_spectral_mod.spectral_smooth(
                    window_size=window_size, order=order)

            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_smooth,
                               dtype=self.io.defaults.envi_write.dtype,
                               force=self.io.defaults.envi_write.force,
                               ext=self.io.defaults.envi_write.ext,
                               interleave=self.io.defaults.envi_write.interleave,
                               byteorder=self.io.defaults.envi_write.byteorder,
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
            self._write_stats(dir_out, df_smooth_stats, fname_csv=name_append[1:] + '-stats.csv')
            # return df_smooth_stats

    def _execute_spec_smooth_pp(self, fname, base_dir_out, folder_name,
                                name_append, window_size, order, stats):
        '''
        Actually executes the spectral smooth to keep the main function a bit
        cleaner
        '''
        print('\nSpectrally smoothing: {0}'.format(fname))
        if stats is True:
            df_smooth_stats = pd.DataFrame(
                    columns=['fname', 'mean', 'std', 'cv'])

        self.io.read_cube(fname)
        self.my_spectral_mod = spec_mod(self.io.spyfile)
        base_dir = os.path.dirname(fname)
        if base_dir_out is None:
            dir_out, name_append = self._save_file_setup(
                    base_dir, folder_name, name_append)
        else:
            dir_out, name_append = self._save_file_setup(
                    base_dir_out, folder_name, name_append)
        name_print = self._get_name_print()
        array_smooth, metadata = self.my_spectral_mod.spectral_smooth(
                window_size=window_size, order=order)

        name_label = (name_print + name_append + '.' +
                      self.io.defaults.envi_write.interleave)
        metadata['label'] = name_label

        hdr_file = os.path.join(dir_out, name_label + '.hdr')
        self.io.write_cube(hdr_file, array_smooth,
                           dtype=self.io.defaults.envi_write.dtype,
                           force=self.io.defaults.envi_write.force,
                           ext=self.io.defaults.envi_write.ext,
                           interleave=self.io.defaults.envi_write.interleave,
                           byteorder=self.io.defaults.envi_write.byteorder,
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
            return df_smooth_stats

    def _get_fname_similar(self, name_to_match, base_dir, search_ext='bip',
                           level=0):
        '''
        Gets a similar filename from another directory
        '''
        fname_list = self._recurs_dir(base_dir, search_ext=search_ext,
                                      level=level)
        fname_similar = []
        for fname in fname_list:
            f = os.path.split(fname)[-1]
            fname_short = f[:f.find('-')]
            # if name_to_match in fname:
            if name_to_match == fname_short:
                fname_similar.append(fname)
        msg1 = ('No files found with a similar name to "{0}". Please be '
                'sure the images are created before continuing (e.g., did '
                'you perform band math yet?)\n\nbase_dir: {1}'
                ''.format(name_to_match, base_dir))
        msg2 = ('Multiple files found with a similar name to {0}. Please '
                'delete files that are not relevant to continue.\n\nbase_dir: '
                '{1}'.format(name_to_match, base_dir))
        assert len(fname_similar) > 0, msg1
        assert len(fname_similar) == 1, msg2
        return fname_similar[0]

    def _get_array_similar(self, dir_search):
        '''
        Retrieves the array from a directory with a similar name to the loaded
        datacube (i.e., there must be a datacube loaded; self.io.spyfile should
        not be ``None``; compares to ``self.io.name_short``).

        Parameters:
            dir_search: directory to search
        '''
        msg = ('Please load a SpyFile prior to using this function')
        assert self.io.spyfile is not None, msg
        if not os.path.isdir(dir_search):
            msg = ('The passed directory does not exist; please pass a valid '
                   'directory path.\nDirectory: {0}'.format(dir_search))
            raise IOError(msg)
        fname_similar = self._get_fname_similar(
                self.io.name_short, dir_search,
                search_ext=self.io.defaults.envi_write.interleave, level=0)
        fpath_similar = os.path.join(dir_search, fname_similar)
        io_mask = hsio()
        io_mask.read_cube(fpath_similar)
        array = io_mask.spyfile.load()
        metadata = io_mask.spyfile.metadata
        return array, metadata

    def _get_class_mask(self, row, filter_cols, n_classes=1):
        '''
        Finds the class with the lowest NDVI in ``row`` and returns the class ID
        to be used to dictate which pixels get masked

        Parameters:
            n_classes (``int``): number of classes to mask; if 1, then will mask
            the minimum ndvi; if more than 1, all classes (default: 1)
        '''
        row_ndvi = row[filter_cols].astype(float)
        row_ndvi = row_ndvi.dropna()
        print(row_ndvi)
        print(n_classes)
        if len(row_ndvi) == n_classes:
            n_classes -= 1
        row_ndvi_small = row_ndvi.nsmallest(n=n_classes)
        class_name = row_ndvi_small.index.values.tolist()
        class_mask = []
        for name in class_name:
            class_int = int(re.search(r'\d+', name).group())
            class_mask.append(class_int)
        return class_mask

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
            out_files: include the full pathname, filename, and ext of all
                files that have ``search_exp`` in their name.
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

    def _save_file_setup(self, base_dir_out, folder_name, name_append):
        '''
        Basic setup items when saving manipulated image files to disk

        Parameters:
            base_dir_out (``str``): Parent directory that all processed datacubes
                will be saved.
            folder_name (``str`` or ``None``): Folder to add to
                ``base_dir_out`` to save all the processed datacubes.
            name_append (``str``): name to append to the filename.
        '''
#        if base_dir_out is None:
#            base_dir_out = os.path.join(self.base_dir, folder_name)
        if pd.isnull(folder_name):
            folder_name = ''
        dir_out = os.path.join(base_dir_out, folder_name)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        if name_append is None:
            name_append = ''
        else:
            if name_append[0] != '-':
                name_append = '-' + str(name_append)
        return dir_out, name_append

    def _get_name_label(self, row, name_print, name_append):
        '''
        Gets the name label for the datacube. By doing this before the
        processing operation, we can check to see if the datacube has
        already been processed and pass over if desired.
        '''
        if 'study' in row.index:
            if row['study'] is not None:
                name_study = 'study_' + str(row['study'] + '_')
        else:
            name_study = ''
        if 'date' in row.index:
            if row['date'] is not None:
                if isinstance(row['date'], str):
                    row['date'] = pd.to_datetime(row['date'])
                name_date = ('date_' + str(row['date'].year).zfill(4) +
                             str(row['date'].month).zfill(2) +
                             str(row['date'].day).zfill(2) + '_')
        else:
            name_date = ''
        if row['plot_id_ref'] is not None:
            name_plot = 'plot_' + str(row['plot_id_ref'])
        else:
            name_plot = ''
        if ((len(name_study) >= 1) and (len(name_date) >= 1) and
            (len(name_plot) >= 1)):  # then remove the name_print variable
            name_label = (name_study + name_date + name_plot + name_append +
                          '.' + self.io.defaults.envi_write.interleave)
        else:
            name_label1 = (name_print + '_' + name_study + name_date +
                           name_plot)
            if name_label1[-1] == '_':
                name_label1 = name_label1[:-1]
            name_label = (name_label1 + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
        return name_label

    def _get_name_print(self, fname_in=None):
        '''

        '''
        name_print = self.io.name_short
        if name_print is None and fname_in is not None:
            base_name = os.path.basename(fname_in)
            name_print = base_name[:base_name.find('-', base_name.rfind('_'))]
        msg = ('Could not get a name for input datacube.\n')
        assert name_print is not None, msg
        return name_print

    def _read_spectra_from_file(self, fname, columns):
        '''
        Reads a single spectra from file
        '''
        self.io.read_spec(fname + '.hdr')
        meta_bands = self.io.tools.meta_bands
        array = self.io.spyfile_spec.load()
        data = list(np.reshape(array, (array.shape[2])) * 100)
        data.insert(0, self.io.name_plot)
        data.insert(0, os.path.basename(fname))
        df_spec_file = pd.DataFrame(data=[data], columns=columns)
        return df_spec_file

    def _print_progress(self, iteration, total, prefix='', suffix='',
                       decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = f'{"█" * filled_length}{"-" * (bar_length - filled_length)}'
        sys.stdout.write(f'\r{prefix} |{bar}| {percents}% {suffix}'),
        # sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix))
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def cube_to_spectra(self, fname_list=None, base_dir=None, search_ext='bip',
                        dir_level=0, base_dir_out=None,
                        folder_name='cube_to_spec',
                        name_append='cube-to-spec',
                        write_geotiff=True, out_dtype=False, out_force=None,
                        out_ext=False, out_interleave=False,
                        out_byteorder=False):
        '''
        Calculates the mean and standard deviation for each cube in
        ``fname_list`` and writes the result to a ".spec" file.

        Parameters:
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``, and
                ``dir_level`` parameters for files to process (default: ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir``
            folder_name (``str``): folder to add to ``base_dir_out`` to save all
                the processed datacubes (default: 'cube_to_spec').
            name_append (``str``): name to append to the filename (default:
                'cube-to-spec').
            write_geotiff (``bool``): whether to save the masked RGB image as a
                geotiff alongside the masked datacube.
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults``, and are
                therefore accessible at a high level. See
                ``hsio.set_io_defaults()`` for more information on each of the
                settings.

        Note:
            The following ``batch`` example builds on the API example results
            of the `spatial_mod.crop_many_gdf`_ function. Please complete the
            `spatial_mod.crop_many_gdf`_ example to be sure your directory
            (i.e., ``base_dir``) is populated with multiple hyperspectral
            datacubes. The following example will be using datacubes located in
            the following directory:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf``

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip')  # searches for all files in ``base_dir`` with a ".bip" file extension

            Use ``batch.cube_to_spectra`` to calculate the *mean* and *standard
            deviation* across all pixels for each of the datacubes in
            ``base_dir``.

            >>> hsbatch.cube_to_spectra(base_dir=base_dir, write_geotiff=False, out_force=True)
            Calculating mean spectra: F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1011.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\cube_to_spec\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1011-cube-to-spec-mean.spec
            Calculating mean spectra: F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1012.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\cube_to_spec\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1012-cube-to-spec-mean.spec
            Calculating mean spectra: F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1013.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\cube_to_spec\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1013-cube-to-spec-mean.spec
            ...

            Use ``seaborn`` to visualize the spectra of plots 1011, 1012, and
            1013. Notice how ``hsbatch.io.name_plot`` is utilized to retrieve
            the plot ID, and how the *"history"* tag is referenced from the
            metadata to determine the number of pixels whose reflectance was
            averaged to create the mean spectra. Also remember that pixels
            across the original input image likely represent a combination of
            soil, vegetation, and shadow.

            >>> import seaborn as sns
            >>> import re
            >>> fname_list = [r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\cube_to_spec\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1011-cube-to-spec-mean.spec',
                              r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\cube_to_spec\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1012-cube-to-spec-mean.spec',
                              r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\cube_to_spec\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1013-cube-to-spec-mean.spec']
            >>> colors = ['red', 'green', 'blue']
            >>> for fname, color in zip(fname_list, colors):
            >>>     hsbatch.io.read_spec(fname)
            >>>     meta_bands = list(hsbatch.io.tools.meta_bands.values())
            >>>     data = hsbatch.io.spyfile_spec.load().flatten() * 100
            >>>     hist = hsbatch.io.spyfile_spec.metadata['history']
            >>>     pix_n = re.search('<pixel number: (.*)>', hist).group(1)
            >>>     ax = sns.lineplot(x=meta_bands, y=data, color=color, label='Plot '+hsbatch.io.name_plot+' (n='+pix_n+')')
            >>> ax.set_xlabel('Wavelength (nm)', weight='bold')
            >>> ax.set_ylabel('Reflectance (%)', weight='bold')
            >>> ax.set_title(r'API Example: `batch.cube_to_spectra`', weight='bold')

            .. image:: ../img/batch/cube_to_spectra.png

        .. _spatial_mod.crop_many_gdf: hs_process.spatial_mod.html#hs_process.spatial_mod.crop_many_gdf
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        append_extra = '-mean'
        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append,
                                               append_extra, ext='.spec')

        pb_i = 0
        pb_len = len(fname_list)
        pb_prefix = 'cube_to_spectra:'
        self._print_progress(pb_i, pb_len, prefix=pb_prefix)

        fname_list_p = tqdm(fname_list) if self.progress_bar is True else fname_list
        for idx, fname in enumerate(fname_list_p):
            if self.progress_bar is True:
                fname_list_p.set_description('Processing file {0}/{1}'.format(idx, len(fname_list)))
            self.io.read_cube(fname)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            name_label = (name_print + name_append + append_extra + '.' +
                          self.io.defaults.envi_write.interleave)
            if self._file_exists_check(
                    dir_out, name_label, write_geotiff=write_geotiff,
                    write_spec=True) is True:
                self.print_progress(idx+1, pb_len, prefix=pb_prefix)
                continue

            # print('Calculating mean spectra: {0}'.format(fname))
            spec_mean, spec_std, array = self.io.tools.mean_datacube(
                    self.io.spyfile)
            metadata = self.io.spyfile.metadata.copy()
            # because this is specialized, we should make our own history str
            n_pix = self.io.spyfile.nrows * self.io.spyfile.ncols
            hist_str = (' -> hs_process.batch.cube_to_spectra[<pixel number: '
                        '{0}>]'.format(n_pix))
            metadata['history'] += hist_str
            name_label_spec = (os.path.splitext(name_label)[0] +
                               '.spec')
            if write_geotiff is True:
                self._write_geotiff(array, fname, dir_out, name_label,
                                    metadata, self.io.tools)
            # Now write spec (will change map info on metadata)
            self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
                             metadata)
            # self._print_progress(idx+1, pb_len, prefix=pb_prefix)

    def segment_composite_band(self, fname_list=None, base_dir=None,
                               search_ext='bip', dir_level=0, base_dir_out=None,
                               folder_name='composite_band',
                               name_append='composite-band',
                               geotiff=True, wl1=None, b1=None,
                               list_range=True, plot_out=True,
                               out_dtype=False, out_force=None, out_ext=False,
                               out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to create a composite band on multiple datacubes
        in the same way. ``batch.segment_composite_band`` is typically used
        prior to  ``batch.segment_create_mask`` to generate the
        images/directory required for the masking process.

        Parameters:
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                band math index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of bands)
                to be used as the first parameter of the band math index; if
                ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            list_range (``bool``): Whether bands/wavelengths passed as a list
                is interpreted as a range of bands (``True``) or for each
                individual band in the list (``False``). If ``list_range`` is
                ``True``, ``b1``/``wl1`` and ``b2``/``wl2`` should be lists
                with two items, and all bands/wavelegths between the two values
                will be used (default: ``True``).
            plot_out (``bool``): whether to save a histogram of the band math
                result (default: ``True``).
            geotiff (``bool``): whether to save the masked RGB image as a
                geotiff alongside the masked datacube.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        # checks filenames
        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            append_extra = '-comp-{0}'.format(int(np.mean(wl1)))
            fname_list = self._check_processed(
                fname_list, base_dir_out, folder_name, name_append, append_extra)

        self._execute_composite_band(fname_list, base_dir_out, folder_name,
                                     name_append, geotiff, wl1, b1,
                                     list_range, plot_out)

    def segment_band_math(self, fname_list=None, base_dir=None,
                          search_ext='bip', dir_level=0, base_dir_out=None,
                          folder_name='band_math', name_append='band-math',
                          write_geotiff=True, method='ndi', wl1=None, wl2=None,
                          wl3=None, b1=None, b2=None, b3=None,
                          list_range=True, plot_out=True,
                          out_dtype=False, out_force=None, out_ext=False,
                          out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to perform band math on multiple datacubes in the
        same way. ``batch.segment_band_math`` is typically used prior to
        ``batch.segment_create_mask`` to generate the images/directory required
        for the masking process.

        Parameters:
            method (``str``): Must be one of "ndi" (normalized difference
                index), "ratio" (simple ratio index), "derivative"
                (deriviative-type index), or "mcari2" (modified chlorophyll
                absorption index2). Indicates what kind of band math should be
                performed on the input datacube. The "ndi" method leverages
                ``segment.band_math_ndi()``, the "ratio" method leverages
                ``segment.band_math_ratio()``, and the "derivative" method
                leverages ``segment.band_math_derivative()``. Please see the
                ``segment`` documentation for more information (default:
                "ndi").
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                band math index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl2 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                band math index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of bands) to be
                used as the first parameter of the band math index;
                if ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            b2 (``int``, ``float``, or ``list``): the band (or set of bands) to be
                used as the second parameter of the band math
                index; if ``list``, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: ``None``).
            list_range (``bool``): Whether bands/wavelengths passed as a list is
                interpreted as a range of bands (``True``) or for each individual
                band in the list (``False``). If ``list_range`` is ``True``,
                ``b1``/``wl1`` and ``b2``/``wl2`` should be lists with two items, and
                all bands/wavelegths between the two values will be used
                (default: ``True``).
            plot_out (``bool``): whether to save a histogram of the band math
                result (default: ``True``).
            write_geotiff (``bool``): whether to save the masked RGB image as a
                geotiff alongside the masked datacube.

        Note:
            The following ``batch`` example builds on the API example results
            of the `spatial_mod.crop_many_gdf`_ function. Please complete the
            `spatial_mod.crop_many_gdf`_ example to be sure your directory
            (i.e., ``base_dir``) is populated with multiple hyperspectral
            datacubes. The following example will be using datacubes located in
            the following directory:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf``

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip')  # searches for all files in ``base_dir`` with a ".bip" file extension

            Use ``batch.segment_band_math`` to compute the MCARI2 (Modified
            Chlorophyll Absorption Ratio Index Improved; Haboudane et al.,
            2004) spectral index for each of the datacubes in ``base_dir``. See
            `Harris Geospatial`_ for more information about the MCARI2 spectral
            index and references to other spectral indices.

            >>> folder_name = 'band_math_mcari2-800-670-550'  # folder name can be modified to be more descriptive in what type of band math is being performed
            >>> method = 'mcari2'  # must be one of "ndi", "ratio", "derivative", or "mcari2"
            >>> wl1 = 800
            >>> wl2 = 670
            >>> wl3 = 550
            >>> hsbatch.segment_band_math(base_dir=base_dir, folder_name=folder_name,
                                          name_append='band-math', write_geotiff=True,
                                          method=method, wl1=wl1, wl2=wl2, wl3=wl3,
                                          plot_out=True, out_force=True)
            Bands used (``b1``): [198]
            Bands used (``b2``): [135]
            Bands used (``b3``): [77]
            Wavelengths used (``b1``): [799.0016]
            Wavelengths used (``b2``): [669.6752]
            Wavelengths used (``b3``): [550.6128]
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\band_math_mcari2-800-670-550\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1011-band-math-mcari2-800-670-550.bip
            ...

            ``batch.segment_band_math`` creates a new folder in ``base_dir``
            (in this case the new directory is
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\band_math_mcari2-800-670-550``)
            which contains several data products. The **first** is
            ``band-math-stats.csv``: a spreadsheet containing summary
            statistics for each of the image cubes that were processed via
            ``batch.segment_band_math``; stats include *pixel count*,
            *mean*, *standard deviation*, *median*, and *percentiles* across
            all image pixels.

            **Second** is a ``geotiff`` file for each of the image cubes after the
            band math processing. This can be opened in *QGIS* to visualize in
            a spatial reference system, or can be opened using any software
            that supports floating point *.tif* files.

            .. image:: ../img/batch/segment_band_math_plot_611-band-math-mcari2-800-670-550_tif.png

            **Third** is the band math raster saved in the *.hdr* file format.
            Note that the data conained here should be the same as in the
            *.tif* file, so it's a matter of preference as to what may be more
            useful. This single band *.hdr* can also be opend in *QGIS*.

            **Fourth** is a histogram of the band math data contained in the
            image. The histogram illustrates the 90th percentile value, which
            may be useful in the segmentation step (e.g., see
            `batch.segment_create_mask`_).

            .. image:: ../img/batch/segment_band_math_plot_611-band-math-mcari2-800-670-550.png

        .. _Harris Geospatial: https://www.harrisgeospatial.com/docs/NarrowbandGreenness.html#Modified3
        .. _batch.segment_create_mask: hs_process.batch.html#hs_process.batch.segment_create_mask
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        # else fname_list must be passed directly

        if method == 'ndi':
            append_extra = ('-{0}-{1}-{2}'
                            ''.format(method, int(np.mean(wl1)),
                                      int(np.mean(wl2))))
        elif method == 'ratio':
            append_extra = ('-{0}-{1}-{2}'
                            ''.format(method, int(np.mean(wl1)),
                                      int(np.mean(wl2))))
        elif method == 'derivative':
            append_extra = ('-{0}-{1}-{2}-{3}'
                            ''.format(method, int(np.mean(wl1)),
                                      int(np.mean(wl2)),
                                      int(np.mean(wl3))))
        elif method == 'mcari2':
            append_extra = ('-{0}-{1}-{2}-{3}'
                            ''.format(method, int(np.mean(wl1)),
                                      int(np.mean(wl2)),
                                      int(np.mean(wl3))))

        # checks filenames
        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(
                fname_list, base_dir_out, folder_name, name_append, append_extra)
        self._execute_band_math(fname_list, base_dir_out, folder_name,
                                name_append, write_geotiff, method, wl1, wl2,
                                wl3, b1, b2, b3, list_range, plot_out)

    def segment_create_mask(self, fname_list=None, base_dir=None,
                            search_ext='bip', dir_level=0, mask_dir=None,
                            base_dir_out=None,
                            folder_name='mask', name_append='mask',
                            write_datacube=True, write_spec=True,
                            write_geotiff=True, mask_thresh=None,
                            mask_percentile=None, mask_side='lower',
                            out_dtype=False, out_force=None, out_ext=False,
                            out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to create a masked array on many datacubes.
        ``batch.segment_create_mask`` is typically used after
        ``batch.segment_band_math`` to mask all the datacubes in a directory
        based on the result of the band math process.

        Parameters:
            mask_thresh (``float`` or ``int``): The value for which to mask the
                array; should be used with ``side`` parameter (default: ``None``).
            mask_percentile (``float`` or ``int``): The percentile of pixels to
                mask; if ``percentile``=95 and ``side``='lower', the lowest 95% of
                pixels will be masked following the band math operation
                (default: ``None``; range: 0-100).
            mask_side (``str``): The side of the threshold for which to apply the
                mask. Must be either 'lower', 'upper', 'outside', or ``None``;
                if 'lower', everything below the threshold will be masked; if
                'outside', the ``thresh`` / ``percentile`` parameter must be
                list-like with two values indicating the lower and upper bounds
                - anything outside of these values will be masked out; if
                ``None``, only the values that exactly match the threshold will
                be masked (default: 'lower').
            geotiff (``bool``): whether to save the masked RGB image as a geotiff
                alongside the masked datacube.

        Note:
            The following ``batch`` example builds on the API example results
            of `spatial_mod.crop_many_gdf`_ and `batch.segment_band_math`_.
            Please complete each of those API examples to be sure your
            directories (i.e., ``base_dir``, and ``mask_dir``) are populated
            with image files. The following example will be masking datacubes
            located in:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf``
            based on MCARI2 images located in:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\\band_math_mcari2-800-670-550``

        Example:
            Load and initialize the ``batch`` module, ensuring ``base_dir`` is
            a valid directory

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip')  # searches for all files in ``base_dir`` with a ".bip" file extension

            There must be a single-band image that will be used to determine
            which datacube pixels are to be masked (determined via the
            ``mask_dir`` parameter). Point to the directory that contains the
            MCARI2 images.

            >>> mask_dir = os.path.join(base_dir, 'band_math_mcari2-800-670-550')
            >>> print(os.path.isdir(mask_dir))
            True

            Indicate how the MCARI2 images should be used to determine which
            hyperspectal pixels are to be masked. The available parameters for
            controlling this are ``mask_thresh``, ``mask_percentile``, and
            ``mask_side``. We will mask out all pixels that fall below the
            MCARI2 90th percentile.

            >>> mask_percentile = 90
            >>> mask_side = 'lower'

            Finally, indicate the folder to save the masked datacubes and
            perform the batch masking via ``batch.segment_create_mask``

            >>> folder_name = 'mask_mcari2_90th'
            >>> hsbatch.segment_create_mask(base_dir=base_dir, mask_dir=mask_dir,
                                            folder_name=folder_name,
                                            name_append='mask-mcari2-90th', geotiff=True,
                                            mask_percentile=mask_percentile,
                                            mask_side=mask_side)
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1011-mask-mcari2-90th.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1011-mask-mcari2-90th-spec-mean.spec
            ...

            .. image:: ../img/batch/segment_create_mask_inline.png

            ``batch.segment_create_mask`` creates a new folder in ``base_dir``
            named according to the ``folder_name`` parameter
            (in this case the new directory is
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th``)
            which contains several data products. The **first** is
            ``mask-stats.csv``: a spreadsheet containing the band math
            threshold value for each image file. In this example, the MCARI2
            value corresponding to the 90th percentile is listed.

            +------------+------------+-------------+
            | fname      | plot_id    |lower-pctl-90|
            +============+============+=============+
            | ...        | 1011       | 0.83222     |
            +------------+------------+-------------+
            | ...        | 1012       | 0.81112     |
            +------------+------------+-------------+
            | ...        | 1013       | 0.74394     |
            +------------+------------+-------------+

            ...etc.

            **Second** is a ``geotiff`` file for each of the image cubes after the
            masking procedure. This can be opened in *QGIS* to visualize in
            a spatial reference system, or can be opened using any software
            that supports floating point *.tif* files. The masked pixels are
            saved as ``null`` values and should render transparently.

            .. image:: ../img/batch/segment_create_mask_geotiff.png

            **Third** is the full hyperspectral datacube, also with the masked
            pixels saved as ``null`` values. Note that the only pixels
            remaining are the 10% with the highest MCARI2 values.

            .. image:: ../img/batch/segment_create_mask_datacube.png

            **Fourth** is the mean spectra across the unmasked datacube pixels.
            This is illustrated above by the green line plot (the light green
            shadow represents the standard deviation for each band).

        .. _Harris Geospatial: https://www.harrisgeospatial.com/docs/NarrowbandGreenness.html#Modified3
        .. _batch.segment_band_math: hs_process.batch.html#hs_process.batch.segment_band_math
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
        self._execute_mask(fname_list, mask_dir, base_dir_out, folder_name,
                           name_append, write_datacube, write_spec,
                           write_geotiff, mask_thresh, mask_percentile,
                           mask_side)

    def spatial_crop(self, fname_sheet=None, base_dir=None, search_ext='bip',
                     dir_level=0, base_dir_out=None,
                     folder_name='spatial_crop', name_append='spatial-crop',
                     write_geotiff=True, method='single', gdf=None, out_dtype=False,
                     out_force=None, out_ext=False, out_interleave=False,
                     out_byteorder=False):
        '''
        Iterates through a spreadsheet that provides necessary information
        about how each image should be cropped and how it should be saved.

        If ``gdf`` is passed (a geopandas.GoeDataFrame polygon file), the
        cropped images will be shifted to the center of appropriate 'plot_id'
        polygon.

        Parameters:
            fname_sheet (``fname``, ``pandas.DataFrame``, or ``None``, optional):
                The filename of the spreadsheed that provides the
                necessary information for fine-tuning the batch process
                cropping. See below for more information about the required and
                optional contents of ``fname_sheet`` and how to properly format
                it. Optionally, ``fname_sheet`` can be a ``Pandas.DataFrame``.
                If left to ``None``, ``base_dir`` and ``gdf`` must be passed.
            base_dir (``str``, optional): directory path to search for files to
                spatially crop; if ``fname_sheet`` is not ``None``,
                ``base_dir`` will be ignored (default: ``None``).
            base_dir_out (``str``, optional): output directory of the cropped
                image (default: ``None``).
            folder_name (``str``, optional): folder to add to ``base_dir_out``
                to save all the processed datacubes (default: 'spatial_crop').
            name_append (``str``, optional): name to append to the filename
                (default: 'spatial-crop').
            write_geotiff (``bool``, optional): whether to save an RGB image as
                a geotiff alongside the cropped datacube.
            method (``str``, optional): Must be one of "single" or
                "many_gdf". Indicates whether a single plot should be cropped
                from the input datacube or if many/multiple plots should be
                cropped from the input datacube. The "single" method leverages
                `spatial_mod.crop_single()`_ and the "many_gdf" method
                leverages `spatial_mod.crop_many_gdf()`_. Please
                see the ``spatial_mod`` documentation for more information
                (default: "single").
            gdf (``geopandas.GeoDataFrame``, optional): the plot names and
                polygon geometery of each of the plots; 'plot_id' must be used as
                a column name to identify each of the plots, and should be an
                integer.
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults``, and are
                therefore accessible at a high level. See
                `hsio.set_io_defaults()`_ for more information on each of the
                settings.

        **Tips and Tricks for** ``fname_sheet`` **when** ``gdf`` **is not passed**

        If ``gdf`` is not passed, ``fname_sheet`` may have the following
        required column headings that correspond to the relevant parameters in
        `spatial_mod.crop_single()`_ and `spatial_mod.crop_many_gdf()`_:

        #. "directory"
        #. "name_short"
        #. "name_long"
        #. "ext"
        #. "pix_e_ul"
        #. "pix_n_ul".

        With this minimum input, ``batch.spatial_crop`` will read in each
        image, crop from the upper left pixel (determined as
        ``pix_e_ul``/``pix_n_ul``) to the lower right pixel calculated
        based on ``crop_e_pix``/``crop_n_pix`` (which is the width of the
        cropped area in units of pixels).

        Note:
            ``crop_e_pix`` and ``crop_n_pix`` have default values (see
            `defaults.crop_defaults()`_), but they can also be passed
            specifically for each datacube by including appropriate columns in
            ``fname_sheet`` (which takes precedence over
            ``defaults.crop_defaults``).

        ``fname_sheet`` may also have the following optional column headings:

        #. "crop_e_pix"
        #. "crop_n_pix"
        #. "crop_e_m"
        #. "crop_n_m"
        #. "buf_e_pix"
        #. "buf_n_pix"
        #. "buf_e_m"
        #. "buf_n_m"
        #. "gdf_shft_e_m"
        #. "gdf_shft_n_m"
        #. "plot_id_ref"
        #. "study"
        #. "date"

        **More** ``fname_sheet`` **Tips and Tricks**

        #. These optional inputs passed via ``fname_sheet`` allow more control
           over exactly how the images are to be cropped. For a more detailed
           explanation of the information that many of these columns are
           intended to contain, see the documentation for
           `spatial_mod.crop_single()`_ and `spatial_mod.crop_many_gdf()`_.
           Those parameters not referenced should be apparent in the API
           examples and tutorials.

        #. If the column names are different in ``fname_sheet`` than described
           here, `defaults.spat_crop_cols()`_ can be modified to indicate which
           columns correspond to the relevant information.

        #. The *date* and *study* columns do not impact how the datacubes are
           to be cropped, but if this information exists,
           ``batch.spatial_crop`` adds it to the filename of the cropped
           datacube. This can be used to avoid overwriting datacubes with
           similar names, and is especially useful when processing imagery from
           many dates and/or studies/locations and saving them in the same
           directory. This information is appended to the end of the
           ``hsio.name_short`` string. An example filename is
           *plot_9_3_pika_gige_1_study_wells_date_20180628_plot_527-spatial-crop.bip*.

        #. Any other columns can be added to ``fname_sheet``, but
           ``batch.spatial_crop()`` does not use them in any way.

        Note:
            The following ``batch`` example only actually processes *a single*
            hyperspectral image. If more datacubes were present in
            ``base_dir``, however, ``batch.spatial_crop`` would process all
            datacubes that were available.

        Note:
            This example uses ``spatial_mod.crop_many_gdf`` to crop many
            plots from a datacube using a polygon geometry file describing the
            spatial extent of each plot.

        Example:

            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> import geopandas as gpd
            >>> import pandas as pd
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip', dir_level=0,
                                progress_bar=True)  # searches for all files in ``base_dir`` with a ".bip" file extension

            Load the plot geometry as a ``geopandas.GeoDataFrame``

            >>> fname_gdf = r'F:\\nigo0024\Documents\hs_process_demo\plot_bounds.geojson'
            >>> gdf = gpd.read_file(fname_gdf)

            Perform the spatial cropping using the *"many_gdf"* ``method``.
            Note that nothing is being bassed to ``fname_sheet`` here, so
            ``batch.spatial_crop`` is simply going to attempt to crop all plots
            contained within ``gdf`` that overlap with any datacubes in
            ``base_dir``. This option does not allow for any flexibility
            regarding minor adjustments to the cropping procedure (e.g.,
            offset to the plot location in the datacube relative to the
            location in the ``gdf``), but it is the most straightforward way to
            run ``batch.spatial_crop`` because it does not depend on anything
            to be passed to ``fname_sheet``. It does, however, allow you to
            adjust the plot buffer relative to ``gdf`` via
            ``hsbatch.io.defaults.crop_defaults``

            >>> hsbatch.io.defaults.crop_defaults.buf_e_m = 2
            >>> hsbatch.io.defaults.crop_defaults.buf_n_m = 0.5
            >>> hsbatch.io.set_io_defaults(force=True)
            >>> hsbatch.spatial_crop(base_dir=base_dir, method='many_gdf',
                                     gdf=gdf)
            Spatially cropping: F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_1018-spatial-crop.bip
            Spatially cropping: F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_918-spatial-crop.bip

            .. image:: ../img/batch/spatial_crop_inline.png

            A new folder was created in ``base_dir``
            - ``F:\\nigo0024\Documents\hs_process_demo\spatial_crop`` - that
            contains the cropped datacubes and the cropped ``geotiff`` images.
            The Plot ID from the ``gdf`` is used to name each datacube
            according to its plot ID. The ``geotiff`` images can be opened in
            *QGIS* to visualize the images after cropping them.

            .. image:: ../img/batch/spatial_crop_tifs.png

            The cropped images were brightened in *QGIS* to emphasize the
            cropped boundaries. The plot boundaries are overlaid for reference
            (notice the 2.0 m buffer on the East/West ends and the 0.5 m buffer
            on the North/South sides).

        .. _defaults.crop_defaults(): hs_process.defaults.html#hs_process.defaults.crop_defaults
        .. _defaults.spat_crop_cols(): hs_process.defaults.html#hs_process.defaults.spat_crop_cols
        .. _hsio.set_io_defaults(): hs_process.hsio.html#hs_process.hsio.set_io_defaults
        .. _spatial_mod.crop_single(): hs_process.spatial_mod.html#hs_process.spatial_mod.crop_single
        .. _spatial_mod.crop_many_gdf(): hs_process.spatial_mod.html#hs_process.spatial_mod.crop_many_gdf
        '''
        if method == 'many_gdf':
            msg1 = ('Please pass a valid ``geopandas.GeoDataFrame`` if using '
                    'the "many_gdf" method.\n')
            msg2 = ('Please be sure the passed ``geopandas.GeoDataFrame`` has '
                    'a column by the name of "plot_id", indicating the plot '
                    'ID for each polygon geometry if using the "many_gdf" '
                    'method.\n')
            assert isinstance(gdf, gpd.GeoDataFrame), msg1
            assert 'plot_id' in gdf.columns, msg2
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)

        if fname_sheet is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_sheet is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        else:  # fname_list comes from fname_sheet
            fname_list = None
        # Either fname_sheet or fname_list should be None
        self._execute_crop(fname_sheet, fname_list, base_dir_out,
                           folder_name, name_append, write_geotiff, method, gdf)

    def spectra_combine(self, fname_list=None, base_dir=None,
                        search_ext='bip', dir_level=0, base_dir_out=None,
                        out_dtype=False, out_force=None, out_ext=False,
                        out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to gather all pixels from every image in a
        directory, compute the mean and standard deviation, and save as a
        single spectra (i.e., a spectra file is equivalent to a single spectral
        pixel with no spatial information).

        Parameters:
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``,
                and ``dir_level`` parameters for files to process (default:
                ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``,
                ``base_dir`` will be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir`` (default:
                ``None``).
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults, and are
                therefore accessible at a high level. See
                ``hsio.set_io_defaults()`` for more information on each of the
                settings.

        Note:
            The following example will load in several small hyperspectral
            radiance datacubes *(not reflectance)* that were previously cropped
            manually (via Spectronon software). These datacubes represent the
            radiance values of grey reference panels that were placed in the
            field to provide data necessary for converting radiance imagery
            to reflectance. These particular datacubes were extracted
            from several different images captured within ~10 minutes of each
            other.

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\cube_ref_panels'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir)

            Combine all the *radiance* datacubes in the directory via
            ``batch.spectra_combine``.

            >>> hsbatch.spectra_combine(base_dir=base_dir, search_ext='bip',
                                        dir_level=0)
            Combining datacubes/spectra into a single mean spectra.
            Number of input datacubes/spectra: 7
            Total number of pixels: 1516
            Saving F:\\nigo0024\Documents\hs_process_demo\cube_ref_panels\spec_mean_spy.spec

            Visualize the combined spectra by opening in *Spectronon*. The
            solid line represents the mean radiance spectra across all pixels
            and images in ``base_dir``, and the lighter, slightly transparent
            line represents the standard deviation of the radiance across all
            pixels and images in ``base_dir``.

            .. image:: ../img/batch/spectra_combine.png

            Notice the lower signal at the oxygen absorption region (near 770
            nm). After converting datacubes to reflectance, it may be
            desireable to spectrally clip this region (see
            `spec_mod.spectral_clip()`_)

        .. _spec_mod.spectral_clip(): hs_process.spec_mod.html#hs_process.spec_mod.spectral_clip
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            msg1 = ('``base_dir`` is not a valid directory.\n')
            assert os.path.isdir(base_dir), msg1
            self.base_dir = base_dir
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg2 = ('Please set ``fname_list`` or ``base_dir`` to indicate '
                    'which datacubes should be processed.\n')
            assert base_dir is not None, msg2
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        self._execute_spec_combine(fname_list, base_dir_out)

    def spectra_to_csv(self, fname_list=None, base_dir=None, search_ext='spec',
                       dir_level=0, base_dir_out=None, name='stats-spectra',
                       multithread=False):
        '''
        Reads all the ``.spec`` files in a direcory and saves their reflectance
        information to a ``.csv``. ``batch.spectra_to_csv`` is identical to
        ``batch.spectra_to_df`` except a ``.csv`` file is saved rather than
        returning a ``pandas.DataFrame``.

        Parameters:
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``, and
                ``dir_level`` parameters for files to process (default: ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, file is saved to ``base_dir``
            name (``str``): The output filename (default: "stats-spectra").
            multithread (``bool``): Whether to leverage multi-thread processing
                when reading the .spec files. Setting to ``True`` should speed
                up the time it takes to read all .spec files.

        Note:
            The following example builds on the API example results of
            `batch.segment_band_math()`_ and `batch.segment_create_mask()_.
            Please complete each of those API examples to be sure your
            directory (i.e.,
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th``)
            is populated with image files.

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir)

            Read all the ``.spec`` files in ``base_dir`` and save them to a
            ``.csv`` file.

            >>> hsbatch.spectra_to_csv(base_dir=base_dir, search_ext='spec',
                                       dir_level=0)
            Writing mean spectra to a .csv file.
            Number of input datacubes/spectra: 40
            Output file location: F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th\stats-spectra.csv

            When ``stats-spectra.csv`` is opened in Microsoft Excel, we can see
            that each row is a ``.spec`` file from a different plot, and each
            column is a particular spectral band/wavelength.

            .. image:: ../img/batch/spectra_to_csv.png

        .. _batch.segment_band_math(): hs_process.batch.html#hs_process.batch.segment_band_math
        .. _batch.segment_create_mask(): hs_process.batch.html#hs_process.batch.segment_create_mask
        '''
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        if base_dir_out is None:
            fname_csv = os.path.join(base_dir, name + '.csv')
        else:
            fname_csv = os.path.join(base_dir_out, name + '.csv')
        print('Writing mean spectra to a .csv file.\n'
              'Number of input datacubes/spectra: {0}\nOutput file location: '
              '{1}'.format(len(fname_list), fname_csv))
        # df_spec = None
        # for fname in fname_list:
        #     self.io.read_spec(fname + '.hdr')
        #     meta_bands = self.io.tools.meta_bands
        #     array = self.io.spyfile_spec.load()
        #     data = list(np.reshape(array, (array.shape[2])) * 100)
        #     data.insert(0, self.io.name_plot)
        #     data.insert(0, os.path.basename(fname))
        #     if df_spec is None:
        #         columns = list(meta_bands.values())
        #         columns.insert(0, 'wavelength')
        #         columns.insert(0, np.nan)
        #         bands = list(meta_bands.keys())
        #         bands.insert(0, 'plot_id')
        #         bands.insert(0, 'fname')
        #         df_spec = pd.DataFrame(data=[bands], columns=columns)
        #     df_spec_temp = pd.DataFrame(data=[data], columns=columns)
        #     df_spec = df_spec.append(df_spec_temp)

        # load the data from the Spectral Python (SpyFile) object
        self.io.read_spec(fname_list[0] + '.hdr')  # read first file to build df_spec column headings
        meta_bands = self.io.tools.meta_bands
        columns = list(meta_bands.values())
        columns.insert(0, 'wavelength')
        columns.insert(0, np.nan)
        bands = list(meta_bands.keys())
        bands.insert(0, 'plot_id')
        bands.insert(0, 'fname')
        df_spec = pd.DataFrame(data=[bands], columns=columns)

        # if multithread is True:
        #     with ThreadPoolExecutor() as executor:  # defaults to min(32, os.cpu_count() + 4)
        #         future_df_spec = {
        #             executor.submit(self._read_spectra_from_file,
        #                             fname,
        #                             df_spec.columns): fname for fname in fname_list}
        #         for future in as_completed(future_df_spec):
        #             data = future_df_spec[future]
        #             try:
        #                 df_spec_file = future.result()
        #                 df_spec = df_spec.append(df_spec_file)
        #             except Exception as exc:
        #                 print('%r generated an exception: %s' % (data, exc))
        # else:
        for fname in fname_list:
            df_spec_file = self._read_spectra_from_file(fname, df_spec.columns)
            df_spec = df_spec.append(df_spec_file)
        df_spec.to_csv(fname_csv, index=False)

    def spectra_to_df(self, fname_list=None, base_dir=None, search_ext='spec',
                      dir_level=0, multithread=False):
        '''
        Reads all the .spec files in a direcory and returns their data as a
        ``pandas.DataFrame`` object. ``batch.spectra_to_df`` is identical to
        ``batch.spectra_to_csv`` except a ``pandas.DataFrame`` is returned
        rather than saving a ``.csv`` file.

        Parameters:
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``, and
                ``dir_level`` parameters for files to process (default: ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            multithread (``bool``): Whether to leverage multi-thread processing
                when reading the .spec files. Setting to ``True`` should speed
                up the time it takes to read all .spec files.

        Note:
            The following example builds on the API example results of
            `batch.segment_band_math()`_ and `batch.segment_create_mask()_.
            Please complete each of those API examples to be sure your
            directory (i.e.,
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th``)
            is populated with image files.

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\mask_mcari2_90th'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir)

            Read all the ``.spec`` files in ``base_dir`` and load them to
            ``df_spec``, a ``pandas.DataFrame``.

            >>> df_spec = hsbatch.spectra_to_df(base_dir=base_dir, search_ext='spec',
                                                dir_level=0)
            Writing mean spectra to a ``pandas.DataFrame``.
            Number of input datacubes/spectra: 40

            When visualizing ``df_spe`` in `Spyder`_, we can see that each row
            is a ``.spec`` file from a different plot, and each column is a
            particular spectral band.

            .. image:: ../img/batch/spectra_to_df.png

            It is somewhat confusing to conceptualize spectral data by band
            number (as opposed to the wavelenth it represents).
            ``hs_process.hs_tools.get_band`` can be used to retrieve
            spectral data for all plots via indexing by wavelength. Say we need
            to access reflectance at 710 nm for each plot.

            >>> df_710nm = df_spec[['fname', 'plot_id', hsbatch.io.tools.get_band(710)]]

            .. image:: ../img/batch/spectra_to_df_710nm.png

        .. _batch.segment_band_math(): hs_process.batch.html#hs_process.batch.segment_band_math
        .. _batch.segment_create_mask(): hs_process.batch.html#hs_process.batch.segment_create_mask
        .. _Spyder: https://www.spyder-ide.org/
        '''
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        print('Writing mean spectra to a ``pandas.DataFrame``.\n'
              'Number of input datacubes/spectra: {0}'
              ''.format(len(fname_list)))

        # load the data from the Spectral Python (SpyFile) object
        # df_spec = None
        # for fname in fname_list:
        #     self.io.read_spec(fname + '.hdr')
        #     meta_bands = self.io.tools.meta_bands
        #     array = self.io.spyfile_spec.load()
        #     data = list(np.reshape(array, (array.shape[2])))
        #     data.insert(0, self.io.name_plot)
        #     data.insert(0, os.path.basename(fname))
        #     if df_spec is None:
        #         bands = list(meta_bands.keys())
        #         bands.insert(0, 'plot_id')
        #         bands.insert(0, 'fname')
        #         df_spec = pd.DataFrame(columns=bands)
        #     df_spec_temp = pd.DataFrame(data=[data], columns=bands)
        #     df_spec = df_spec.append(df_spec_temp)


        # read first file to build df_spec column headings
        self.io.read_spec(fname_list[0] + '.hdr')
        meta_bands = self.io.tools.meta_bands
        bands = list(meta_bands.keys())
        bands.insert(0, 'plot_id')
        bands.insert(0, 'fname')
        df_spec = pd.DataFrame(columns=bands)
        # if multithread is True:
        #     with ThreadPoolExecutor() as executor:  # defaults to min(32, os.cpu_count() + 4)
        #         future_df_spec = {
        #             executor.submit(self._read_spectra_from_file,
        #                             fname,
        #                             df_spec.columns): fname for fname in fname_list}
        #         for future in as_completed(future_df_spec):
        #             data = future_df_spec[future]
        #             try:
        #                 df_spec_file = future.result()
        #                 df_spec = df_spec.append(df_spec_file)
        #             except Exception as exc:
        #                 print('%r generated an exception: %s' % (data, exc))
        # else:
        for fname in fname_list:
            df_spec_file = self._read_spectra_from_file(fname, df_spec.columns)
            df_spec = df_spec.append(df_spec_file)

        try:
            df_spec['plot_id'] = pd.to_numeric(df_spec['plot_id'])
        except ValueError:
            print('Unable to convert "plot_id" column to numeric type.\n')
        return df_spec.reset_index(drop=True)

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
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``, and
                ``dir_level`` parameters for files to process (default: ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir``
            folder_name (``str``): folder to add to ``base_dir_out`` to save all
                the processed datacubes (default: 'spec-clip').
            name_append (``str``): name to append to the filename (default:
                'spec-clip').
            wl_bands (``list`` or ``list of lists``): minimum and maximum
                wavelenths to clip from image; if multiple groups of
                wavelengths should be cut, this should be a list of lists. For
                example, wl_bands=[760, 776] will clip all bands greater than
                760.0 nm and less than 776.0 nm;
                wl_bands = [[0, 420], [760, 776], [813, 827], [880, 1000]]
                will clip all band less than 420.0 nm, bands greater than 760.0
                nm and less than 776.0 nm, bands greater than 813.0 nm and less
                than 827.0 nm, and bands greater than 880 nm (default).
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults, and are
                therefore accessible at a high level. See
                ``hsio.set_io_defaults()`` for more information on each of the
                settings.

        Note:
            The following ``batch`` example builds on the API example results
            of the `batch.spatial_crop`_ function. Please complete the
            `batch.spatial_crop`_ example to be sure your directory
            (i.e., ``base_dir``) is populated with multiple hyperspectral
            datacubes. The following example will be using datacubes located in
            the following directory:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_crop``

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_crop'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip')  # searches for all files in ``base_dir`` with a ".bip" file extension

            Use ``batch.spectral_clip`` to clip all spectral bands below
            *420 nm* and above *880 nm*, as well as the bands near the oxygen
            absorption (i.e., *760-776 nm*) and water absorption
            (i.e., *813-827 nm*) regions.

            >>> hsbatch.spectral_clip(base_dir=base_dir, folder_name='spec_clip',
                                      wl_bands=[[0, 420], [760, 776], [813, 827], [880, 1000]])
            Processing 40 files. If this is not what is expected, please check if files have already undergone processing. If existing files should be overwritten, be sure to set the ``out_force`` parameter.
            Spectrally clipping: F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_1011-spatial-crop.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\spec_clip\Wells_rep2_20180628_16h56m_pika_gige_7_1011-spec-clip.bip
            Spectrally clipping: F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_1012-spatial-crop.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\spec_clip\Wells_rep2_20180628_16h56m_pika_gige_7_1012-spec-clip.bip
            ...

            Use ``seaborn`` to visualize the spectra of a single pixel in one
            of the processed images.

            >>> import seaborn as sns
            >>> fname = os.path.join(base_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spatial-crop.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem = hsbatch.io.spyfile.open_memmap()  # datacube before clipping
            >>> meta_bands = list(hsbatch.io.tools.meta_bands.values())
            >>> fname = os.path.join(base_dir, 'spec_clip', 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spec-clip.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem_clip = hsbatch.io.spyfile.open_memmap()  # datacube after clipping
            >>> meta_bands_clip = list(hsbatch.io.tools.meta_bands.values())
            >>> ax = sns.lineplot(x=meta_bands, y=spy_mem[26][29], label='Before spectral clipping', linewidth=3)
            >>> ax = sns.lineplot(x=meta_bands_clip, y=spy_mem_clip[26][29], label='After spectral clipping', ax=ax)
            >>> ax.set_xlabel('Wavelength (nm)', weight='bold')
            >>> ax.set_ylabel('Reflectance (%)', weight='bold')
            >>> ax.set_title(r'API Example: `batch.spectral_clip`', weight='bold')

            .. image:: ../img/batch/spectral_clip_plot.png

            Notice the spectral areas that were clipped, namely the oxygen and
            water absorption regions (~770 and ~820 nm, respectively). There
            is perhaps a lower *signal:noise* ratio in these regions, which was
            the merit for clipping those bands out.

        .. _batch.spatial_crop: hs_process.batch.html#hs_process.batch.spatial_crop
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)

        self._execute_spec_clip(fname_list, base_dir_out, folder_name,
                                name_append, wl_bands)

        # with ThreadPoolExecutor(max_workers=None) as executor:  # defaults to min(32, os.cpu_count() + 4)
        #     future_to_clip = {
        #         executor.submit(self._execute_spec_clip_pp,
        #                         fname,
        #                         base_dir_out, folder_name, name_append, wl_bands): fname for fname in fname_list}


    def spectral_mimic(
            self, fname_list=None, base_dir=None, search_ext='bip',
            dir_level=0, base_dir_out=None, folder_name='spec_mimic',
            name_append='spec-mimic', sensor='sentinel-2a',
            df_band_response=None, col_wl='wl_nm', center_wl='peak',
            out_dtype=False, out_force=None, out_ext=False,
            out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to spectrally mimic a multispectral sensor for
        multiple datacubes in the same way.

        Parameters:
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``,
                and ``dir_level`` parameters for files to process (default:
                ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally resample; if ``fname_list`` is not ``None``,
                ``base_dir`` will be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir``
            folder_name (``str``): folder to add to ``base_dir_out`` to save
                all the processed datacubes (default: 'spec_bin').
            name_append (``str``): name to append to the filename (default:
                'spec-bin').
            sensor (``str``): Should be one of
                ["sentera_6x", "micasense_rededge_3", "sentinel-2a",
                "sentinel-2b", "custom"]; if "custom", ``df_band_response``
                and ``col_wl`` must be passed.
            df_band_response (``pd.DataFrame``): A DataFrame that contains the
                transmissivity (%) for each sensor band (as columns) mapped to
                the continuous wavelength values (as rows). Required if
                ``sensor`` is  "custom", ignored otherwise.
            col_wl (``str``): The column of ``df_band_response`` denoting the
                wavlengths (default: 'wl_nm').
            center_wl (``str``): Indicates how the center wavelength of each
                band is determined. If ``center_wl`` is "peak", the point at
                which transmissivity is at its maximum is used as the center
                wavelength. If ``center_wl`` is "weighted", the weighted
                average is used to compute the center wavelength. Must be one
                of ["peak", "weighted"] (``default: "peak"``).
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults, and are
                therefore accessible at a high level. See
                ``hsio.set_io_defaults()`` for more information on each of the
                settings.

        Note:
            The following ``batch`` example builds on the API example results
            of the `batch.spatial_crop`_ function. Please complete the
            `batch.spatial_crop`_ example to be sure your directory
            (i.e., ``base_dir``) is populated with multiple hyperspectral
            datacubes. The following example will be using datacubes located in
            the following directory:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_crop``

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_crop'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip', progress_bar=True)  # searches for all files in ``base_dir`` with a ".bip" file extension

            Use ``batch.spectral_mimic`` to spectrally mimic the Sentinel-2A
            multispectral satellite sensor.

            >>> hsbatch.spectral_mimic(
                base_dir=base_dir, folder_name='spec_mimic',
                name_append='sentinel-2a',
                sensor='sentinel-2a', center_wl='weighted')
            Processing 40 files. If existing files should be overwritten, be sure to set the ``out_force`` parameter.
            Processing file 39/40: 100%|██████████| 40/40 [00:04<00:00,  8.85it/s]

            Use ``seaborn`` to visualize the spectra of a single pixel in one
            of the processed images.

            >>> import seaborn as sns
            >>> fname = os.path.join(base_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spatial-crop.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem = hsbatch.io.spyfile.open_memmap()  # datacube before mimicking
            >>> meta_bands = list(hsbatch.io.tools.meta_bands.values())
            >>> fname = os.path.join(base_dir, 'spec_mimic', 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-sentinel-2a.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem_sen2a = hsbatch.io.spyfile.open_memmap()  # datacube after mimicking
            >>> meta_bands_sen2a = list(hsbatch.io.tools.meta_bands.values())
            >>> ax = sns.lineplot(x=meta_bands, y=spy_mem[26][29], label='Hyperspectral (Pika II)', linewidth=3)
            >>> ax = sns.lineplot(x=meta_bands_sen2a, y=spy_mem_sen2a[26][29], label='Sentinel-2A "mimic"', marker='o', ms=6, ax=ax)
            >>> ax.set_xlabel('Wavelength (nm)', weight='bold')
            >>> ax.set_ylabel('Reflectance (%)', weight='bold')
            >>> ax.set_title(r'API Example: `batch.spectral_mimic`', weight='bold')

            .. image:: ../img/batch/spectral_mimic_sentinel-2a_plot.png

        .. _batch.spatial_crop: hs_process.batch.html#hs_process.batch.spatial_crop
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate '
                   'which datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)

        self._execute_spec_mimic(fname_list, base_dir_out, folder_name,
                                 name_append, sensor, df_band_response,
                                 col_wl, center_wl)

    def spectral_resample(
            self, fname_list=None, base_dir=None, search_ext='bip',
            dir_level=0, base_dir_out=None, folder_name='spec_bin',
            name_append='spec-bin', bandwidth=None, bins_n=None,
            out_dtype=False, out_force=None, out_ext=False,
            out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to spectrally resample (a.k.a. "bin") multiple
        datacubes in the same way.

        Parameters:
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``,
                and ``dir_level`` parameters for files to process (default:
                ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally resample; if ``fname_list`` is not ``None``,
                ``base_dir`` will be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir``
            folder_name (``str``): folder to add to ``base_dir_out`` to save
                all the processed datacubes (default: 'spec_bin').
            name_append (``str``): name to append to the filename (default:
                'spec-bin').
            bandwidth (``float`` or ``int``): The bandwidth of the bands
                after spectral resampling is complete (units should be
                consistent with that of the .hdr file). Setting ``bandwidth``
                to 10 will consolidate bands that fall within every 10 nm
                interval.
            bins_n (``int``): The number of bins (i.e., "bands") to achieve
                after spectral resampling is complete. Ignored if ``bandwidth``
                is not ``None``.
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults, and are
                therefore accessible at a high level. See
                ``hsio.set_io_defaults()`` for more information on each of the
                settings.

        Note:
            The following ``batch`` example builds on the API example results
            of the `batch.spatial_crop`_ function. Please complete the
            `batch.spatial_crop`_ example to be sure your directory
            (i.e., ``base_dir``) is populated with multiple hyperspectral
            datacubes. The following example will be using datacubes located in
            the following directory:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_crop``

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_crop'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip', progress_bar=True)  # searches for all files in ``base_dir`` with a ".bip" file extension

            Use ``batch.spectral_resample`` to bin ("group") all spectral bands
            into 20 nm bandwidth bands (from ~2.3 nm bandwidth originally) on
            a per-pixel basis.

            >>> hsbatch.spectral_resample(
                base_dir=base_dir, folder_name='spec_bin',
                name_append='spec-bin-20', bandwidth=20)
            Processing 40 files. If existing files should be overwritten, be sure to set the ``out_force`` parameter.
            Processing file 39/40: 100%|██████████| 40/40 [00:00<00:00, 48.31it/s]
            ...

            Use ``seaborn`` to visualize the spectra of a single pixel in one
            of the processed images.

            >>> import seaborn as sns
            >>> fname = os.path.join(base_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spatial-crop.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem = hsbatch.io.spyfile.open_memmap()  # datacube before resampling
            >>> meta_bands = list(hsbatch.io.tools.meta_bands.values())
            >>> fname = os.path.join(base_dir, 'spec_bin', 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spec-bin-20.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem_bin = hsbatch.io.spyfile.open_memmap()  # datacube after resampling
            >>> meta_bands_bin = list(hsbatch.io.tools.meta_bands.values())
            >>> ax = sns.lineplot(x=meta_bands, y=spy_mem[26][29], label='Hyperspectral (Pika II)', linewidth=3)
            >>> ax = sns.lineplot(x=meta_bands_bin, y=spy_mem_bin[26][29], label='Spectral resample (20 nm)', marker='o', ms=6, ax=ax)
            >>> ax.set_xlabel('Wavelength (nm)', weight='bold')
            >>> ax.set_ylabel('Reflectance (%)', weight='bold')
            >>> ax.set_title(r'API Example: `batch.spectral_resample`', weight='bold')

            .. image:: ../img/batch/spectral_resample-20nm_plot.png

        .. _batch.spatial_crop: hs_process.batch.html#hs_process.batch.spatial_crop
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate '
                   'which datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)

        self._execute_spec_resample(fname_list, base_dir_out, folder_name,
                                    name_append, bandwidth, bins_n)

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
            fname_list (``list``, optional): list of filenames to process; if
                left to ``None``, will look at ``base_dir``, ``search_ext``, and
                ``dir_level`` parameters for files to process (default: ``None``).
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
            base_dir_out (``str``): directory path to save all processed
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir``
            folder_name (``str``): folder to add to ``base_dir_out`` to save all
                the processed datacubes (default: 'spec-smooth').
            name_append (``str``): name to append to the filename (default:
                'spec-smooth').
            window_size (``int``): the length of the window; must be an odd
                integer number (default: 11).
            order (``int``): the order of the polynomial used in the filtering;
                must be less than ``window_size`` - 1 (default: 2).
            stats (``bool``): whether to compute some basic descriptive
                statistics (mean, st. dev., and coefficient of variation) of
                the smoothed data array (default: ``False``)
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults, and are
                therefore accessible at a high level. See
                ``hsio.set_io_defaults()`` for more information on each of the
                settings.

        Note:
            The following ``batch`` example builds on the API example results
            of the `batch.spatial_crop`_ function. Please complete the
            `batch.spatial_crop`_ example to be sure your directory
            (i.e., ``base_dir``) is populated with multiple hyperspectral
            datacubes. The following example will be using datacubes located in
            the following directory:
            ``F:\\nigo0024\Documents\hs_process_demo\spatial_crop``

        Example:
            Load and initialize the ``batch`` module, checking to be sure the
            directory exists.

            >>> import os
            >>> from hs_process import batch
            >>> base_dir = r'F:\\nigo0024\Documents\hs_process_demo\spatial_crop'
            >>> print(os.path.isdir(base_dir))
            True
            >>> hsbatch = batch(base_dir, search_ext='.bip')  # searches for all files in ``base_dir`` with a ".bip" file extension

            Use ``batch.spectral_smooth`` to perform a *Savitzky-Golay*
            smoothing operation on each image/pixel in ``base_dir``. The
            ``window_size`` and ``order`` can be adjusted to achieve desired
            smoothing results.

            >>> hsbatch.spectral_smooth(base_dir=base_dir, folder_name='spec_smooth',
                                        window_size=11, order=2)
            Processing 40 files. If this is not what is expected, please check if files have already undergone processing. If existing files should be overwritten, be sure to set the ``out_force`` parameter.
            Spectrally smoothing: F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_1011-spatial-crop.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\spec_smooth\Wells_rep2_20180628_16h56m_pika_gige_7_1011-spec-smooth.bip
            Spectrally smoothing: F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_1012-spatial-crop.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\spec_smooth\Wells_rep2_20180628_16h56m_pika_gige_7_1012-spec-smooth.bip
            ...

            Use ``seaborn`` to visualize the spectra of a single pixel in one
            of the processed images.

            >>> import seaborn as sns
            >>> fname = os.path.join(base_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spatial-crop.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem = hsbatch.io.spyfile.open_memmap()  # datacube before smoothing
            >>> meta_bands = list(hsbatch.io.tools.meta_bands.values())
            >>> fname = os.path.join(base_dir, 'spec_smooth', 'Wells_rep2_20180628_16h56m_pika_gige_7_1011-spec-smooth.bip')
            >>> hsbatch.io.read_cube(fname)
            >>> spy_mem_clip = hsbatch.io.spyfile.open_memmap()  # datacube after smoothing
            >>> meta_bands_clip = list(hsbatch.io.tools.meta_bands.values())
            >>> ax = sns.lineplot(x=meta_bands, y=spy_mem[26][29], label='Before spectral smoothing', linewidth=3)
            >>> ax = sns.lineplot(x=meta_bands_clip, y=spy_mem_clip[26][29], label='After spectral smoothing', ax=ax)
            >>> ax.set_xlabel('Wavelength (nm)', weight='bold')
            >>> ax.set_ylabel('Reflectance (%)', weight='bold')
            >>> ax.set_title(r'API Example: `batch.spectral_smooth`', weight='bold')

            .. image:: ../img/batch/spectral_smooth_plot.png

            Notice how the *"choppiness"* of the spectral curve is lessened
            after the smoothing operation. There are spectral regions that
            perhaps had a lower *signal:noise* ratio and did not do particularlly
            well at smoothing (i.e., < 410 nm, ~770 nm, and ~820 nm). It may be
            wise to perform ``batch.spectral_smooth`` *after*
            `batch.spectral_clip`_.

        .. _batch.spatial_crop: hs_process.batch.html#hs_process.batch.spatial_crop
        .. _batch.spectral_clip: hs_process.batch.html#hs_process.batch.spectral_clip
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
        self._execute_spec_smooth(
                fname_list, base_dir_out, folder_name, name_append,
                window_size, order, stats)
        # if df_stats is not None:
        #     return df_stats


        # # Parallel threading
        # with ThreadPoolExecutor(max_workers=None) as executor:  # defaults to min(32, os.cpu_count() + 4)
        #     future_to_clip = {
        #         executor.submit(self._execute_spec_smooth_pp,
        #                         fname,
        #                         base_dir_out, folder_name, name_append,
        #                         window_size, order, stats): fname for fname in fname_list}

        #     df_stats = None
        #     if stats == True:
        #         for future in as_completed(future_to_clip):
        #             smooth = future_to_clip[future]
        #             try:
        #                 df_smooth_temp = future.result()
        #                 if df_stats is None:
        #                     df_stats = df_smooth_temp.copy()
        #                 else:
        #                     df_stats = df_stats.append(df_smooth_temp, ignore_index=True)
        #             except Exception as exc:
        #                 print('%r generated an exception: %s' % (smooth, exc))
        #             else:
        #                 print('%r page is %d bytes' % (smooth, len(data)))

        #         base_dir = os.path.dirname(fname_list[0])
        #         if base_dir_out is None:
        #             dir_out, name_append = self._save_file_setup(
        #                     base_dir, folder_name, name_append)
        #         else:
        #             dir_out, name_append = self._save_file_setup(
        #                     base_dir_out, folder_name, name_append)

        #         fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
        #         if os.path.isfile(fname_stats) and self.io.defaults.envi_write.force is False:
        #             df_stats_in = pd.read_csv(fname_stats)
        #             df_smooth_stats = df_stats_in.append(df_stats)
        #         df_smooth_stats.to_csv(fname_stats)
        #         return df_smooth_stats
