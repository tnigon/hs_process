# -*- coding: utf-8 -*-
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from matplotlib import pyplot as plt

from hs_process.utilities import defaults
from hs_process.utilities import hsio
from hs_process.segment import segment
from hs_process.spec_mod import spec_mod
from hs_process.spatial_mod import spatial_mod


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
    def __init__(self, base_dir=None, search_ext='.bip', dir_level=0):
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
        '''
        self.base_dir = base_dir
        self.search_ext = search_ext
        self.dir_level = dir_level
        self.fname_list = None
        if base_dir is not None:
            self.fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        self.io = hsio()
        self.my_spectral_mod = None
        self.my_spatial_mod = None
        self.my_segment = None

    def _try_spat_crop_col_key(self, key, df_row):
        '''
        Gets value of batch.io.defaults.spat_crop_cols[``key``]; returns
        ``None`` if there is a KeyError

        Adds ``key`` to batch.io.defaults.spat_crop_cols if it does not yet
        exist, but then of course the ``value`` that is returned will be
        ``None``
        '''
        if key not in self.io.defaults.spat_crop_cols.keys():
            print(key)
            self.io.defaults.spat_crop_cols[key] = key
        try:
            value = df_row[self.io.defaults.spat_crop_cols[key]]
        except KeyError:  # try to retrieve a default value
            try:
                value = self.io.defaults.crop_defaults[key]
            except KeyError:
                value = None
        return value

    def _check_processed(self, fname_list, base_dir_out, folder_name,
                         name_append, append_extra=None):
        '''
        Checks if any files in fname_list have already (presumably) undergone
        processing. This is determined by checking if a file exists with a
        particular name based on the filename in fname_list and naming
        parameters (i.e,. ``folder_name`` and ``name_append``).
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

            name_label = (name_print + name_append + append_extra + '.' +
                          self.io.defaults.envi_write.interleave)
            if os.path.isfile(os.path.join(dir_out, name_label)):
                fname_list_final.remove(fname)
        msg = ('There are no files to process. Please check if files have '
               'already undergone processing. If existing files should be '
               'overwritten, be sure to set the ``out_force`` parameter.\n')
        assert(len(fname_list_final) > 0), msg
        print('Processing {0} files. If this is not what is expected, please '
              'check if files have already undergone processing. If existing '
              'files should be overwritten, be sure to set the ``out_force`` '
              'parameter.\n'.format(len(fname_list_final)))
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
                'plot_id': self._try_spat_crop_col_key('plot_id', row),
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
        if not pd.notnull(crop_specs['plot_id']):
            crop_specs['plot_id'] = None
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

    def _execute_mask(self, fname_list, mask_dir, base_dir_out, folder_name,
                      name_append, geotiff, mask_thresh, mask_percentile,
                      mask_side):
        '''
        Actually creates the mask to keep the main function a bit cleaner
        '''
        if mask_thresh is not None and mask_percentile is not None:
            type_mask = ('{0}-thresh-{1}-pctl-{2}'.format(
                    mask_side, mask_thresh, mask_percentile))
        elif mask_thresh is not None and mask_percentile is None:
            type_mask = ('{0}-thresh-{1}'.format(
                    mask_side, mask_thresh))
        elif mask_thresh is None and mask_percentile is not None:
            type_mask = ('{0}-pctl-{1}'.format(
                    mask_side, mask_percentile))
        columns = ['fname', 'plot_id', type_mask]
        df_stats = pd.DataFrame(columns=columns)

        for fname in fname_list:
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
            array = self.io.spyfile.load()

            if mask_dir is None:
                mask_dir = os.path.join(self.io.base_dir, 'band_math')
            array_bandmath, metadata_bandmath = self._get_array_similar(
                    mask_dir)
#            array_bandmath = np.ma.masked_array(
#                    array_bandmath, mask=array_kmeans.mask)
            array_mask, metadata_bandmath = self.io.tools.mask_array(
                    array_bandmath, metadata_bandmath, thresh=mask_thresh,
                    percentile=mask_percentile, side=mask_side)

            stat_mask_mean = np.nanmean(array_mask)
            data = [fname, self.io.name_plot, stat_mask_mean]
            df_stats_temp = pd.DataFrame(data=[data], columns=columns)
            df_stats = df_stats.append(df_stats_temp, ignore_index=True)
            spec_mean, spec_std, datacube_masked = self.io.tools.mean_datacube(
                    array, array_mask.mask)
#            metadata = self.io.spyfile.metadata.copy()
            # because this is specialized, we should make our own history str
            hist_str = (" -> hs_process.batch.segment_create_mask[<"
                        "label: 'mask_thresh?' value:{0}; "
                        "label: 'mask_percentile?' value:{1}; "
                        "label: 'mask_side?' value:{2}>]"
                        "".format(mask_thresh, mask_percentile, mask_side))
            metadata['history'] += hist_str
            metadata_geotiff['history'] += hist_str

            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            self._write_datacube(dir_out, name_label, datacube_masked,
                                 metadata)
            name_label_spec = (os.path.splitext(name_label)[0] +
                               '-spec-mean.spec')
            self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
                             metadata)

            self.array_mask = array_mask
            if geotiff is True:
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
        fname_csv = 'mask-stats.csv'
        fname_csv_full = os.path.join(dir_out, fname_csv)
        df_stats.to_csv(fname_csv_full, index=False)
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

    def _execute_band_math(self, fname_list, base_dir_out, folder_name,
                           name_append, geotiff, method, wl1, wl2, wl3, b1, b2,
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
        for fname in fname_list:
            self.io.read_cube(fname)
            dir_out, name_print, name_append = self._band_math_setup(
                    base_dir_out, folder_name, fname, name_append, method)
            self.my_segment = segment(self.io.spyfile)

            if method == 'ndi':
                array_bm, metadata = self.my_segment.band_math_ndi(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}.{3}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        self.io.defaults.envi_write.interleave))
            elif method == 'ratio':
                array_bm, metadata = self.my_segment.band_math_ratio(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}.{3}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        self.io.defaults.envi_write.interleave))
            elif method == 'derivative':
                array_bm, metadata = self.my_segment.band_math_derivative(
                        wl1=wl1, wl2=wl2, wl3=wl3, b1=b1, b2=b2, b3=b3,
                        list_range=list_range, print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}-{3}.{4}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        int(np.mean(wl3)),
                                        self.io.defaults.envi_write.interleave))
            elif method == 'mcari2':
                array_bm, metadata = self.my_segment.band_math_mcari2(
                        wl1=wl1, wl2=wl2, wl3=wl3, b1=b1, b2=b2, b3=b3,
                        list_range=list_range, print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}-{3}.{4}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        int(np.mean(wl3)),
                                        self.io.defaults.envi_write.interleave))

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
            if geotiff is True:
                self._write_geotiff(array_bm, fname, dir_out, name_label,
                                    metadata, self.my_segment.tools)

        fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
        if os.path.isfile(fname_stats) and self.io.defaults.envi_write.force is False:
            df_stats_in = pd.read_csv(fname_stats)
            df_stats = df_stats_in.append(df_stats)
        df_stats.to_csv(fname_stats, index=False)

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
#        elif fname_sheet is not None and fname_list is not None:
#            msg2 = ('Both ``fname_sheet`` and ``fname_list`` were passed. '
#                    '``fname_list`` (perhaps from ``base_dir``) will be '
#                    'ignored.\n')
#            print(msg2)
#            if isinstance(fname_sheet, pd.DataFrame):
#                df_plots = fname_sheet
#            elif os.path.splitext(fname_sheet)[-1] == '.csv':
#                df_plots = pd.read_csv(fname_sheet)
        else:  # fname_list was passed and df_plots will be figured out later
            msg3 = ('``method`` is "single", but ``fname_list`` was passed '
                    'instead of ``fname_sheet``.\n\nIf performing '
                    '``crop_single``, please pass ``fname_sheet``.\n\nIf '
                    'performing ``crop_many_gdf``, please pass ``fname_list`` '
                    '(perhaps via ``base_dir``).\n')
            assert method in ['many_grid', 'many_gdf'], msg3
            return

    def _crop_loop(self, df_plots, gdf, base_dir_out, folder_name,
                   name_append, geotiff):
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
        for idx, row in df_plots.iterrows():
            cs = self._crop_read_sheet(row)
            fname = os.path.join(cs['directory'], cs['fname'])
            print('\nSpatially cropping: {0}'.format(fname))
            name_long = cs['name_long']  # ``None`` if it was never set
            plot_id = cs['plot_id']
            name_short = cs['name_short']
            fname_hdr = fname + '.hdr'
            self.io.read_cube(fname_hdr, name_long=name_long,
                              name_plot=plot_id, name_short=name_short)
            self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
            self.my_spatial_mod.defaults = self.io.defaults
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        cs['directory'], folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            cs = self._pix_to_mapunit(cs)
#            if method == 'single':
            array_crop, metadata = self.my_spatial_mod.crop_single(
                    cs['pix_e_ul'], cs['pix_n_ul'], cs['crop_e_pix'],
                    cs['crop_n_pix'], buf_e_pix=cs['buf_e_pix'],
                    buf_n_pix=cs['buf_n_pix'])
            if row['plot_id'] is not None:
                name_plot = '_' + str(row['plot_id'])
            else:
                name_plot = ''
            name_label = (name_print + name_plot + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            fname = os.path.join(cs['directory'], cs['fname'])
            self._write_datacube(dir_out, name_label, array_crop, metadata)
            if geotiff is True:
                self._write_geotiff(array_crop, fname, dir_out, name_label,
                                    metadata, self.my_spatial_mod.tools)

    def _crop_many_read_row(self, row, gdf, method):
        '''
        Helper function for reading a row of a dataframe with information about
        how to crop an image many times
        '''
        cs = self._crop_read_sheet(row)  # this function creates cs['fname']
        fname_in = os.path.join(cs['directory'], cs['fname'])
        print('Filename: {0}'.format(fname_in))
        name_long = cs['name_long']  # ``None`` if it was never set
        plot_id = cs['plot_id']
        name_short = cs['name_short']
        fname_hdr = fname_in + '.hdr'
        self.io.read_cube(fname_hdr, name_long=name_long,
                          name_plot=plot_id, name_short=name_short)
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
        return df_plots_many

    def _many_grid(self, cs):
        '''Wrapper to get consice access to ``spatial_mod.crop_many_grid()'''
        df_plots = self.my_spatial_mod.crop_many_grid(
            cs['plot_id'], pix_e_ul=cs['pix_e_ul'], pix_n_ul=cs['pix_n_ul'],
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
        if cs['plot_id'] is None:
            cs['plot_id'] = self.io.defaults.crop_defaults.plot_id
        if cs['buf_e_m'] is None:
            cs['buf_e_m'] = self.io.defaults.crop_defaults.buf_e_m
        if cs['buf_n_m'] is None:
            cs['buf_n_m'] = self.io.defaults.crop_defaults.buf_n_m
        if cs['buf_e_pix'] is None:
            cs['buf_e_pix'] = self.io.defaults.crop_defaults.buf_e_pix
        if cs['buf_n_pix'] is None:
            cs['buf_n_pix'] = self.io.defaults.crop_defaults.buf_n_pix

        df_plots = self.my_spatial_mod.crop_many_gdf(
            cs['plot_id'], pix_e_ul=cs['pix_e_ul'], pix_n_ul=cs['pix_n_ul'],
            crop_e_m=cs['crop_e_m'], crop_n_m=cs['crop_n_m'],
            buf_e_m=cs['buf_e_m'], buf_n_m=cs['buf_n_m'],
            n_plots=cs['n_plots'])
        return df_plots

    def _crop_execute(self, fname_sheet, fname_list, base_dir_out, folder_name,
                      name_append, geotiff, method, gdf):
        '''
        Actually executes the spatial crop to keep the main function a bit
        cleaner

        Either `fname_sheet` or `fname_list` should be None
        '''
        df_plots = self._crop_check_input(fname_sheet, fname_list, method)
        if method == 'single':
            self._crop_loop(df_plots)
        elif method == 'many_gdf' and isinstance(df_plots, pd.DataFrame):
            # if user passes a dataframe, just do whatever it says..
            # loop through each row, doing crop_many_gdf() on each row with
            # whatever parameters are passed via the columns..
            # we should assume that each row of df_plots contains an image that
            # should have crop_many_gdf performed on it to create a new
            # dataframe that can be passed to _crop_loop()
            for idx, row in df_plots.iterrows():
                print('\nComputing information to spatially crop via '
                      '``spatial_mod.crop_many_gdf``:')
                df_plots_many = self._crop_many_read_row(row, gdf, method)
                self._crop_loop(df_plots_many, gdf, base_dir_out, folder_name,
                                name_append, geotiff)
        elif method == 'many_gdf' and df_plots is None:
            for fname_in in fname_list:
                self.io.read_cube(fname_in)
                self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
                self.my_spatial_mod.defaults = self.io.defaults
                df_plots_many = self.my_spatial_mod.crop_many_gdf()
                self._crop_loop(df_plots_many, gdf, base_dir_out, folder_name,
                                name_append, geotiff)
        elif method == 'many_grid' and isinstance(df_plots, pd.DataFrame):
            for idx, row in df_plots.iterrows():
                print('\nComputing information to spatially crop via '
                      '``spatial_mod.crop_many_grid``:')
                df_plots_many = self._crop_many_read_row(row, gdf, method)
                self._crop_loop(df_plots_many, gdf, base_dir_out, folder_name,
                                name_append, geotiff)
        else:
            msg = ('Either ``method`` or ``df_plots`` are not defined '
                   'correctly. If using "many_grid" method, please be sure '
                   '``df_plots`` is being populated correcty\n\n``method``: '
                   '{0}'.format(method))
            raise ValueError(msg)

#        for idx, row in df_plots.iterrows():
#            cs = self._crop_read_sheet(row)
#            fname = os.path.join(cs['directory'], cs['fname'])
#            print('\nSpatially cropping: {0}'.format(fname))
#            name_long = cs['name_long']  # ``None`` if it was never set
#            plot_id = cs['plot_id']
#            name_short = cs['name_short']
#            fname_hdr = fname + '.hdr'
#            self.io.read_cube(fname_hdr, name_long=name_long,
#                              name_plot=plot_id, name_short=name_short)
#            cs = self._pix_to_mapunit(cs)
#            self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
#            if base_dir_out is None:
#                dir_out, name_append = self._save_file_setup(
#                        cs['directory'], folder_name, name_append)
#            else:
#                dir_out, name_append = self._save_file_setup(
#                        base_dir_out, folder_name, name_append)
#            name_print = self._get_name_print()
##            if method == 'single':
#            array_crop, metadata = self.my_spatial_mod.crop_single(
#                    cs['pix_e_ul'], cs['pix_n_ul'], cs['crop_e_pix'],
#                    cs['crop_n_pix'], buf_e_pix=cs['buf_e_pix'],
#                    buf_n_pix=cs['buf_n_pix'])
#            if row['plot_id'] is not None:
#                name_plot = '_' + str(row['plot_id'])
#            else:
#                name_plot = ''
#            name_label = (name_print + name_plot + name_append + '.' +
#                          self.io.defaults.envi_write.interleave)
#            fname = os.path.join(cs['directory'], cs['fname'])
#            self._write_datacube(dir_out, name_label, array_crop, metadata)
#            if geotiff is True:
#                self._write_geotiff(array_crop, fname, dir_out, name_label,
#                                    metadata, self.my_spatial_mod.tools)
#            else:
#                if method == 'many_grid':
#                    df_plots = self._many_grid(cs)
#                elif method == 'many_gdf':
#                    df_plots = self._many_gdf(cs)
#
#                for idx, row in df_plots.iterrows():  # actually crop the image
#                    # reload spyfile to my_spatial_mod??
#                    self.io.read_cube(fname_hdr, name_long=name_long,
#                                      name_plot=plot_id, name_short=name_short)
#                    self.my_spatial_mod.load_spyfile(self.io.spyfile)
#                    crop_e_pix = cs['crop_e_pix']
#                    crop_n_pix = cs['crop_n_pix']
#                    if pd.isnull(crop_e_pix):
#                        crop_e_pix = row['crop_e_pix']
#                    if pd.isnull(crop_n_pix):
#                        crop_n_pix = row['crop_n_pix']
#
#                    array_crop, metadata = self.my_spatial_mod.crop_single(
#                        row['pix_e_ul'], row['pix_n_ul'],
#                        crop_e_pix=crop_e_pix,
#                        crop_n_pix=crop_n_pix,
#                        buf_e_pix=cs['buf_e_pix'],
#                        buf_n_pix=cs['buf_n_pix'],
#                        plot_id=row['plot_id'], gdf=gdf)
##                    metadata = row['metadata']
##                    array_crop = row['array_crop']
#                    if row['plot_id'] is not None:
#                        name_plot = '_' + str(row['plot_id'])
#                    else:
#                        name_plot = ''
#                    name_label = (name_print + name_plot + name_append + '.' +
#                                  self.io.defaults.envi_write.interleave)
#                    fname = os.path.join(cs['directory'], cs['fname'])
#                    self._write_datacube(dir_out, name_label, array_crop,
#                                         metadata)
#                    if geotiff is True:
#                        self._write_geotiff(array_crop, fname, dir_out,
#                                            name_label, metadata,
#                                            self.my_spatial_mod.tools)
#                self.metadata = metadata

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
                       tools):
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
                          show_img='inline')
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
        for fname in fname_list:
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

    def _execute_spec_smooth(self, fname_list, base_dir_out, folder_name,
                             name_append, window_size, order, stats):
        '''
        Actually executes the spectral smooth to keep the main function a bit
        cleaner
        '''
        if stats is True:
            df_smooth_stats = pd.DataFrame(
                    columns=['fname', 'mean', 'std', 'cv'])
        for fname in fname_list:
            print('\nSpectrally smoothing: {0}'.format(fname))
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
            fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
            if os.path.isfile(fname_stats) and self.io.defaults.envi_write.force is False:
                df_stats_in = pd.read_csv(fname_stats)
                df_smooth_stats = df_stats_in.append(df_smooth_stats)
            df_smooth_stats.to_csv(fname_stats)
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
            if name_to_match in fname:
                fname_similar.append(fname)
        msg1 = ('No files found with a similar name to {0}. Please be '
                'sure the images are created before continuing (e.g., did '
                'you perform band math yet?)\n\nbase_dir: {1}'
                ''.format(name_to_match, base_dir))
        msg2 = ('Multiple files found with a similar name to {0}. Please '
                'delete files that are not relevant to continue.\n\nbase_dir: '
                '{1}'.format(name_to_match, base_dir))
        assert len(fname_similar) != 0, msg1
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
        fname_kmeans = self._get_fname_similar(
                self.io.name_short, dir_search,
                search_ext=self.io.defaults.envi_write.interleave, level=0)
        fpath_kmeans = os.path.join(dir_search, fname_kmeans)
        io_mask = hsio()
        io_mask.read_cube(fpath_kmeans)
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
            folder_name (``str``): Folder to add to ``base_dir_out`` to save all
                the processed datacubes.
            name_append (``str``): name to append to the filename.
        '''
#        if base_dir_out is None:
#            base_dir_out = os.path.join(self.base_dir, folder_name)
        dir_out = os.path.join(base_dir_out, folder_name)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        if name_append is None:
            name_append = ''
        else:
            if name_append[0] != '-':
                name_append = '-' + str(name_append)
        return dir_out, name_append

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
#
#    def _plot_histogram(self, array, fname_fig, title=None, xlabel=None,
#                        percentile=90, fontsize=16, color='#444444'):
#        '''
#        Plots a histogram with the percentile value labeled
#        '''
#        if isinstance(array, np.ma.core.MaskedArray):
#            array_m = array.compressed()  # allows for accurate percentile calc
#        else:
#            array_m = np.ma.masked_array(array, mask=False)
#            array_m = array_m.compressed()
#
#        pctl = np.nanpercentile(array_m.flatten(), percentile)
#
#        fig, ax = plt.subplots()
#        ax = sns.distplot(array_m.flatten(), bins=50, color='grey')
#        data_x, data_y = ax.lines[0].get_data()
#
#        y_lim = ax.get_ylim()
#        yi = np.interp(pctl, data_x, data_y)
#        ymax = yi/y_lim[1]
#        ax.axvline(pctl, ymax=ymax, linestyle='--', color=color, linewidth=0.5)
#        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
#
#
#        legend_str = ('Percentile ({0}): {1:.3f}'
#                      ''.format(percentile, pctl))
#        ax.annotate(
#            legend_str,
#            xy=(pctl, yi),
#            xytext=(0.97, 0.94),  # loc to place text
#            textcoords='axes fraction',  # placed relative to axes
#            ha='right',  # alignment of text
#            va='top',
#            fontsize=int(fontsize * 0.9),
#            color=color,
#            bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
#                      ec=(0.5, 0.5, 0.5), alpha=0.5),
#            arrowprops=dict(arrowstyle='-|>',
#                            color=color,
#        #                    patchB=el,
#                            shrinkA=0,
#                            shrinkB=0,
#                            connectionstyle='arc3,rad=-0.3',
#                            linestyle='--',
#                            linewidth=0.7))
#        ax.set_title(title, fontweight='bold', fontsize=int(fontsize * 1.1))
#        ax.set_xlabel(xlabel, fontsize=fontsize)
#        ax.set_ylabel('Frequency (%)', fontsize=fontsize)
#        ax.tick_params(labelsize=fontsize)
#        plt.tight_layout()
#        fig.savefig(fname_fig, dpi=300)

    def cube_to_spectra(self, fname_list=None, base_dir=None, search_ext='bip',
                        dir_level=0, base_dir_out=None,
                        folder_name='cube_to_spec',
                        name_append='cube-to-spec',
                        geotiff=True, out_dtype=False, out_force=None,
                        out_ext=False, out_interleave=False,
                        out_byteorder=False):
        '''
        Calculates the mean and standard deviation for each cube in
        ``fname_list`` and writes the result to a .spec file.

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
            geotiff (``bool``): whether to save the masked RGB image as a geotiff
                alongside the masked datacube.
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in ``batch.io.defaults, and are
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

            >>> hsbatch.cube_to_spectra(base_dir=base_dir, geotiff=False, out_force=True)
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
            across the original input image likely represent a combinatoin of
            soil, vegeation, and shadow.

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

        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)

        for fname in fname_list:
            print('\nCalculating mean spectra: {0}'.format(fname))
            self.io.read_cube(fname)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)

            spec_mean, spec_std, array = self.io.tools.mean_datacube(
                    self.io.spyfile)

            name_print = self._get_name_print()
            name_label = (name_print + name_append + '.' +
                          self.io.defaults.envi_write.interleave)
            metadata = self.io.spyfile.metadata.copy()
            # because this is specialized, we should make our own history str
            n_pix = self.io.spyfile.nrows * self.io.spyfile.ncols
            hist_str = (' -> hs_process.batch.cube_to_spectra[<pixel number: '
                        '{0}>]'.format(n_pix))
            metadata['history'] += hist_str
            name_label_spec = (os.path.splitext(name_label)[0] +
                               '-mean.spec')
            if geotiff is True:
                self._write_geotiff(array, fname, dir_out, name_label,
                                    metadata, self.io.tools)
            # Now write spec (will change map info on metadata)
            self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
                             metadata)

    def segment_band_math(self, fname_list=None, base_dir=None,
                          search_ext='bip', dir_level=0, base_dir_out=None,
                          folder_name='band_math', name_append='band-math',
                          geotiff=True, method='ndi', wl1=None, wl2=None,
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
            geotiff (``bool``): whether to save the masked RGB image as a geotiff
                alongside the masked datacube.

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
                                          name_append='band-math', geotiff=True,
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
            may be useful for in the segmentation step (e.g., see
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
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append,
                                               append_extra)
        self._execute_band_math(fname_list, base_dir_out, folder_name,
                                name_append, geotiff, method, wl1, wl2,
                                wl3, b1, b2, b3, list_range, plot_out)

    def segment_create_mask(self, fname_list=None, base_dir=None,
                            search_ext='bip', dir_level=0, mask_dir=None,
                            base_dir_out=None,
                            folder_name='mask', name_append='mask',
                            geotiff=True, mask_thresh=None,
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
            mask_side (``str``): The side of the threshold or percentile for
                which to apply the mask. Must be either 'lower' or 'upper'; if
                'lower', everything below the threshold/percentile will be
                masked (default: 'lower').
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
            | ...        | 1011       | 0.83341     |
            +------------+------------+-------------+
            | ...        | 1012       | 0.81117     |
            +------------+------------+-------------+
            | ...        | 1013       | 0.75025     |
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
            This is illustrated above by the green plot (the light green shadow
            represents the standard deviation for each band).

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
                           name_append, geotiff, mask_thresh,
                           mask_percentile, mask_side)

    def spatial_crop(self, fname_sheet=None, base_dir=None, search_ext='bip',
                     dir_level=0, base_dir_out=None,
                     folder_name='spatial_crop', name_append='spatial-crop',
                     geotiff=True, method='single', gdf=None, out_dtype=False,
                     out_force=None, out_ext=False, out_interleave=False,
                     out_byteorder=False):
        '''
        Iterates through spreadsheet that provides necessary information about
        how each image should be cropped and how it should be saved.

        If ``gdf`` is passed (a geopandas.GoeDataFrame polygon file), the
        cropped images will be shifted to the center of appropriate "plot"
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
            geotiff (``bool``, optional): whether to save an RGB image as a
                geotiff alongside the cropped datacube.
            method (``str``, optional): Must be one of "single" or
                "many_gdf". Indicates whether a single plot should be cropped
                from the input datacube or if many/multiple plots should be
                cropped from the input datacube. The "single" method leverages
                `spatial_mod.crop_single()`_ and the "many_gdf" method
                leverages `spatial_mod.crop_many_gdf()`_. Please
                see the ``spatial_mod`` documentation for more information
                (default: "single").
            gdf (``geopandas.GeoDataFrame``, optional): the plot names and
                polygon geometery of each of the plots; 'plot' must be used as
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
        #. "plot_id"

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

        #. Any other columns can be added to ``fname_sheet``, but
           ``batch.spatial_crop()`` does not use them in any way.

        Note:
            The following ``batch`` example only actually processes *a single*
            hyperspectral image. If more datacubes were present in
            ``base_dir``, however, ``batch.spatial_crop`` would process all
            datacubes that were available.

        Note:
            This example will uses ``spatial_mod.crop_many_gdf`` to crop many
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
            >>> hsbatch = batch(base_dir, search_ext='.bip', dir_level=0)  # searches for all files in ``base_dir`` with a ".bip" file extension

            Load the plot geometry as a ``geopandas.GeoDataFrame``

            >>> fname_gdf = r'F:\\nigo0024\Documents\hs_process_demo\plot_bounds_small\plot_bounds.shp'
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
            Spatially cropping: F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_crop\Wells_rep2_20180628_16h56m_pika_gige_7_1018-spatial-crop.bip
            Spatially cropping: F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip
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
            msg2 = ('Please be sure the passed ``geopandas.GeoDataFrame`` has a '
                    'column by the name of "plot", indicating the plot ID for '
                    'each polygon geometry if using the "many_gdf" method.\n')
            assert isinstance(gdf, gpd.GeoDataFrame), msg1
            assert 'plot' in gdf.columns, msg2
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

        # Either fname_sheet or fname_list should be None
        self._crop_execute(fname_sheet, fname_list, base_dir_out,
                           folder_name, name_append, geotiff, method, gdf)

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
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the ``batch`` object
            base_dir = self.base_dir
            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        self._execute_spec_combine(fname_list, base_dir_out)

    def spectra_to_csv(self, fname_list=None, base_dir=None, search_ext='spec',
                       dir_level=0, base_dir_out=None):
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
                datacubes; if set to ``None``, a folder named according to the
                ``folder_name`` parameter is added to ``base_dir``

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

        fname_csv = os.path.join(base_dir, 'stats-spectra.csv')
        print('Writing mean spectra to a .csv file.\n'
              'Number of input datacubes/spectra: {0}\nOutput file location: '
              '{1}'.format(len(fname_list), fname_csv))

        # load the data from the Spectral Python (SpyFile) object
        df_spec = None
        for fname in fname_list:
            self.io.read_spec(fname + '.hdr')
            meta_bands = self.io.tools.meta_bands
            array = self.io.spyfile_spec.load()
            data = list(np.reshape(array, (array.shape[2])) * 100)
            data.insert(0, self.io.name_plot)
            data.insert(0, os.path.basename(fname))
            if df_spec is None:
                columns = list(meta_bands.values())
                columns.insert(0, 'wavelength')
                columns.insert(0, np.nan)
                bands = list(meta_bands.keys())
                bands.insert(0, 'plot_id')
                bands.insert(0, 'fname')
                df_spec = pd.DataFrame(data=[bands], columns=columns)
            df_spec_temp = pd.DataFrame(data=[data], columns=columns)
            df_spec = df_spec.append(df_spec_temp)
        df_spec.to_csv(fname_csv, index=False)

    def spectra_to_df(self, fname_list=None, base_dir=None, search_ext='spec',
                      dir_level=0):
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
            ``hs_process.hs_tools.get_band`` can be utilized to retrieve
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
        df_spec = None
        for fname in fname_list:
            self.io.read_spec(fname + '.hdr')
            meta_bands = self.io.tools.meta_bands
            array = self.io.spyfile_spec.load()
            data = list(np.reshape(array, (array.shape[2])))
            data.insert(0, self.io.name_plot)
            data.insert(0, os.path.basename(fname))
            if df_spec is None:
                bands = list(meta_bands.keys())
                bands.insert(0, 'plot_id')
                bands.insert(0, 'fname')
                df_spec = pd.DataFrame(columns=bands)
            df_spec_temp = pd.DataFrame(data=[data], columns=bands)
            df_spec = df_spec.append(df_spec_temp)
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
        df_stats = self._execute_spec_smooth(
                fname_list, base_dir_out, folder_name, name_append,
                window_size, order, stats)
        if df_stats is not None:
            return df_stats

#    def combine_kmeans_bandmath(self, fname_sheet, base_dir_out=None,
#                                folder_name='mask_combine',
#                                name_append='mask-combine',
#                                geotiff=True, kmeans_mask_classes=1,
#                                kmeans_filter='mcari2',
#                                mask_percentile=90, mask_side='lower',
#                                out_dtype=False, out_force=None, out_ext=False,
#                                out_interleave=False, out_byteorder=False):
#        '''
#
#        Parameters:
#            fname_sheet (``fname`` or ``Pandas.DataFrame): The filename of the
#                spreadsheed that provides the necessary information for batch
#                process cropping. See below for more information about the
#                required and optional contents of ``fname_sheet`` and how to
#                properly format it. Optionally, ``fname_sheet`` can be a
#                ``Pandas.DataFrame``
#            kmeans_mask_classes (``int``): number of K-means classes to mask from
#                the datacube. By default, the classes with the lowest average
#                spectral value (e.g., NDVI, GNDVI, MCARI2, etc.; based on
#                ``kmeans_filter`` parameter) will be masked (default: 1).
#            kmeans_filter (``str``): the spectral index to base the K-means mask
#                on. Must be one of 'ndvi', 'gndvi', 'ndre', or 'mcari2'. Note
#                that the K-means aglorithm does not use the in its clustering
#                algorithm (default: 'mcari2').
#        Mask steps:
#            1. load kmeans-stats.csv
#            2. for each row, load spyfile based on "fname"
#            3. get class to mask based on:
#                a. find class with min ndvi: min_class = np.nanmin(class_X_ndvi)
#            4. mask all pixels of min_class
#            5. calculate band math (or load images with band math already calculated)
#            6. mask all pixels below threshold/perecentile
#                a. if percentile, use pctl to determine number of pixels to keep
#                b. if pctl = 90, we should keep 10% of total pixels:
#                    i. find total number of pixels
#                    ii. calculate what 10% is (155*46 = 7130); 7130 * .1 = 713
#                    iii. of pixels that are not masked by kmeans, find 713 pixels with highest bandmath
#                c. if thresh, number doesn't matter
#            7. apply the mask to the datacube and save spectra
#        '''
#        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
#                                out_byteorder)
#
#        if isinstance(fname_sheet, pd.DataFrame):
#            df_kmeans = fname_sheet.copy()
#            fname_sheet = 'dataframe passed'
#        elif os.path.splitext(fname_sheet)[1] == '.csv':
#            df_kmeans = pd.read_csv(fname_sheet)
#
#        msg = ('<kmeans_filter> must be one of "ndvi", "gndvi", "ndre", '
#               'or "mcari2".')
#        assert kmeans_filter in ['ndvi', 'gndvi', 'ndre', 'mcari2'], msg
#        filter_str = '_{0}'.format(kmeans_filter)
#        filter_cols = [col for col in df_kmeans.columns if filter_str in col]
#
#        bandmath_pctl_str = '{0}_pctl'.format(kmeans_filter)
#        bandmath_side_str = '{0}_side'.format(kmeans_filter)
#        columns = ['fname', 'kmeans_class', 'kmeans_nonmasked_pct',
#                   bandmath_pctl_str, bandmath_side_str, 'total_nonmasked_pct']
#        df_stats = pd.DataFrame(columns=columns)
#
#        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
#            fname_list = df_kmeans['fname'].tolist()
#            fname_list = self._check_processed(fname_list, base_dir_out,
#                                               folder_name, name_append)
#            df_kmeans = df_kmeans[df_kmeans['plot_id'].isin(fname_list)]
#
#        for idx, row in df_kmeans.iterrows():  # using stats-kmeans.csv
#            class_mask = self._get_class_mask(row, filter_cols,
#                                              n_classes=kmeans_mask_classes)
#            fname = row['fname']
#            self.io.read_cube(fname)
#            if base_dir_out is None:
#                dir_out, name_append = self._save_file_setup(
#                        os.path.dirname(fname), folder_name, name_append)
#            else:
#                dir_out, name_append = self._save_file_setup(
#                        base_dir_out, folder_name, name_append)
#            name_print = self._get_name_print()
#
#            dir_search = os.path.join(self.io.base_dir, 'kmeans')
#            array_kmeans, metadata_kmeans = self._get_array_similar(dir_search)
#            dir_search = os.path.join(self.io.base_dir, 'band_math')
#            array_bandmath, metadata_bandmath = self._get_array_similar(
#                    dir_search)
#
#            array_kmeans, metadata_kmeans = self.io.tools.mask_array(
#                    array_kmeans, metadata_kmeans, thresh=class_mask,
#                    side=None)  # when side=None, masks the exact match
#            kmeans_pct = (100 * (array_kmeans.count() /
#                            (array_kmeans.shape[0]*array_kmeans.shape[1])))
#
#            # by adding the kmeans mask, hstools.mask_array will consider that
#            # mask when masking by bandmath values (applicable for percentile)
#            array_bandmath = np.ma.masked_array(
#                    array_bandmath, mask=array_kmeans.mask)
#            mask_combined, metadata_bandmath = self.io.tools.mask_array(
#                    array_bandmath, metadata_bandmath,
#                    percentile=mask_percentile, side=mask_side)
#            total_pct = (100 * (mask_combined.count() /
#                            (mask_combined.shape[0]*mask_combined.shape[1])))
#            spec_mean, spec_std, datacube_masked = self.io.tools.mean_datacube(
#                    self.io.spyfile, mask_combined)
#
#            data = [os.path.basename(fname), class_mask, kmeans_pct,
#                    mask_percentile, mask_side, total_pct]
#            df_stats_temp = pd.DataFrame(data=[data], columns=columns)
#            df_stats = df_stats.append(df_stats_temp)
#            name_label = (name_print + name_append + '.' +
#                          self.io.defaults.envi_write.interleave)
#            metadata = self.io.spyfile.metadata.copy()
#            # because this is specialized, we should make our own history str
#            hist_str = (" -> hs_process.batch.combine_kmeans_bandmath[<"
#                        "label: 'fname_sheet?' value:{0}; "
#                        "label: 'kmeans_class?' value:{1}; "
#                        "label: 'mask_percentile?' value:{2}; "
#                        "label: 'mask_side?' value:{3}>]"
#                        "".format(fname_sheet, class_mask, mask_percentile,
#                                  mask_side))
#            metadata['history'] += hist_str
#            self._write_datacube(dir_out, name_label, datacube_masked,
#                                 metadata)
#
#            name_label_spec = (os.path.splitext(name_label)[0] +
#                               '-spec-mean.spec')
#            self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
#                             metadata)
#        fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
#        if os.path.isfile(fname_stats) and self.io.defaults.envi_write.force is False:
#            df_stats_in = pd.read_csv(fname_stats)
#            df_stats = df_stats_in.append(df_stats)
#        df_stats.to_csv(fname_stats, index=False)

#    def segment_kmeans(self, fname_list=None, base_dir=None, search_ext='bip',
#                       dir_level=0, base_dir_out=None, folder_name='kmeans',
#                       name_append='kmeans', geotiff=True,
#                       n_classes=3, max_iter=100, plot_out=True,
#                       out_dtype=False, out_force=None,
#                       out_ext=False, out_interleave=False,
#                       out_byteorder=False):
#        '''
#        Batch processing tool to perform kmeans clustering on multiple
#        datacubes in the same way (uses Spectral Python kmeans tool).
#
#        Parameters:
#            n_classes (``int``): number of classes to cluster (default: 3).
#            max_iter (``int``): maximum iterations before terminating process
#                (default: 100).
#            plot_out (``bool``): whether to save a line plot of the spectra for
#                each class (default: ``True``).
#            geotiff (``bool``): whether to save the masked RGB image as a geotiff
#                alongside the masked datacube.
#        '''
#        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
#                                out_byteorder)
#        if fname_list is None and base_dir is not None:
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#        elif fname_list is None and base_dir is None:
#            # base_dir may have been stored to the ``batch`` object
#            base_dir = self.base_dir
#            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
#                   'datacubes should be processed.\n')
#            assert base_dir is not None, msg
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#
#        if self.io.defaults.envi_write.force is False:  # otherwise just overwrites if it exists
#            fname_list = self._check_processed(fname_list, base_dir_out,
#                                               folder_name, name_append)
#        self._execute_kmeans(fname_list, base_dir_out, folder_name,
#                             name_append, geotiff, n_classes, max_iter,
#                             plot_out)
#
##        if fname_list is not None:
##            self._execute_kmeans(fname_list, base_dir_out, folder_name,
##                                 name_append, geotiff, n_classes, max_iter,
##                                 plot_out, mask_soil=False)
##        elif base_dir is not None:
##            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
##            self._execute_kmeans(fname_list, base_dir_out, folder_name,
##                                 name_append, geotiff, n_classes, max_iter,
##                                 plot_out, mask_soil=False)
##        else:  # fname_list and base_dir are both ``None``
##            # base_dir may have been stored to the ``batch`` object
##            base_dir = self.base_dir
##            msg = ('Please set ``fname_list`` or ``base_dir`` to indicate which '
##                   'datacubes should be processed.\n')
##            assert base_dir is not None, msg
##            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
##            self._execute_kmeans(fname_list, base_dir_out, folder_name,
##                                 name_append, geotiff, n_classes, max_iter,
##                                 plot_out, mask_soil=False)
