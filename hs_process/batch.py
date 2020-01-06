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
    Parent class for batch processing hyperspectral image data
    '''
    def __init__(self, base_dir=None, search_ext='.bip', dir_level=0):
        '''
        Parameters:
            base_dir (`str`, optional): directory path to search for files to
                spectrally clip; if `fname_list` is not `None`, `base_dir` will
                be ignored (default: `None`).
            search_ext (`str`): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if `fname_list` is not `None`, `search_ext` will
                be ignored (default: 'bip').
            dir_level (`int`): The number of directory levels to search; if
                `None`, searches all directory levels (default: 0).
        '''
        self.base_dir = base_dir
        self.search_ext = search_ext
        self.dir_level = dir_level
        self.fname_list = None
        if base_dir is not None:
            self.fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        self.io = hsio()
        self.defaults = defaults.spat_crop_cols
        self.my_spectral_mod = None
        self.my_spatial_mod = None
        self.my_segment = None

    def _try_dict(self, key, df_row):
        if key not in self.defaults.keys():
            print(key)
            self.defaults[key] = key
        try:
            value = df_row[self.defaults[key]]
        except KeyError:
            value = None
        return value

    def _check_processed(self, fname_list, base_dir_out, folder_name,
                         name_append, append_extra=None):
        '''
        Checks if any files in fname_list have already (presumably) undergone
        processing. This is determined by checking if a file exists with a
        particular name based on the filename in fname_list and naming
        parameters (i.e,. `folder_name` and `name_append`).
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
                          self.io.defaults.interleave)
            if os.path.isfile(os.path.join(dir_out, name_label)):
                fname_list_final.remove(fname)
        msg = ('There are no files to process. Please check if files have '
               'already undergone processing. If existing files should be '
               'overwritten, be sure to set the `out_force` parameter.\n')
        assert(len(fname_list_final) > 0), msg
        print('Processing {0} files. If this is not what is expected, please '
              'check if files have already undergone processing. If existing '
              'files should be overwritten, be sure to set the `out_force` '
              'parameter.\n'.format(len(fname_list_final)))
        return fname_list_final

    def _crop_read_sheet(self, row):
        '''
        Reads the necessary information from the spreadsheet and saves it
        to a dictionary

        If this function causes an error, try checking `batch.defaults` - these
        should be adjusted according to the default column names of the input
        (i.e., `fname_sheet`).
        '''
        crop_specs = {
                'directory': self._try_dict('directory', row),
                'fname': self._try_dict('fname', row),
                'name_short': self._try_dict('name_short', row),
                'name_long': self._try_dict('name_long', row),
                'ext': self._try_dict('ext', row),
                'pix_e_ul': self._try_dict('pix_e_ul', row),
                'pix_n_ul': self._try_dict('pix_n_ul', row),
                'plot_id': self._try_dict('plot_id', row),
                'alley_size_e_m': self._try_dict('alley_size_e_m', row),
                'alley_size_n_m': self._try_dict('alley_size_n_m', row),
                'alley_size_e_pix': self._try_dict('alley_size_e_pix', row),
                'alley_size_n_pix': self._try_dict('alley_size_n_pix', row),
                'buf_e_m': self._try_dict('buf_e_m', row),
                'buf_n_m': self._try_dict('buf_n_m', row),
                'buf_e_pix': self._try_dict('buf_e_pix', row),
                'buf_n_pix': self._try_dict('buf_n_pix', row),
                'crop_e_m': self._try_dict('crop_e_m', row),
                'crop_n_m': self._try_dict('crop_n_m', row),
                'crop_e_pix': self._try_dict('crop_e_pix', row),
                'crop_n_pix': self._try_dict('crop_n_pix', row),
                'n_plots_x': self._try_dict('n_plots_x', row),
                'n_plots_y': self._try_dict('n_plots_y', row),
                'n_plots': self._try_dict('n_plots', row)}
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
            if col_name not in self.defaults.keys():
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
        Looks over specifications of `crop_specs`, and converts betweeen pixel
        units and map units if one is populated and the other is `None`
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

    def _many_grid(self, cs):
        '''Wrapper to get consice access to `spatial_mod.crop_many_grid()'''
        df_plots = self.my_spatial_mod.crop_many_grid(
            cs['plot_id'], pix_e_ul=cs['pix_e_ul'], pix_n_ul=cs['pix_n_ul'],
            crop_e_m=cs['crop_e_m'], crop_n_m=cs['crop_n_m'],
            alley_size_n_m=cs['alley_size_n_m'], buf_e_m=cs['buf_e_m'],
            buf_n_m=cs['buf_n_m'], n_plots_x=cs['n_plots_x'],
            n_plots_y=cs['n_plots_y'])
        return df_plots

    def _many_gdf(self, cs):
        '''
        Wrapper to get consice access to `spatial_mod.crop_many_gdf();
        `my_spatial_mod` already has access to `spyfile` and `gdf`, so no need
        to pass them here.
        '''
        df_plots = self.my_spatial_mod.crop_many_gdf(
            cs['plot_id'], pix_e_ul=cs['pix_e_ul'], pix_n_ul=cs['pix_n_ul'],
            crop_e_m=cs['crop_e_m'], crop_n_m=cs['crop_n_m'],
            buf_e_m=cs['buf_e_m'], buf_n_m=cs['buf_n_m'],
            n_plots=cs['n_plots'])
        return df_plots

    def _band_math_setup(self, base_dir_out, folder_name, fname, name_append,
                         method):
        '''
        '''
        msg = ('`method` must be one of either "ndi", "ratio", "derivative", '
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
                          self.io.defaults.interleave)
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
#            metadata['interleave'] = self.io.defaults.interleave
#            name_label_bm = (name_print + name_append + '-{0}-{1}-{2}.'
#                             ''.format(method, int(np.mean(wl1)),
#                                       int(np.mean(wl2))) +
#                             self.io.defaults.interleave)
#            meta_bm['label'] = name_label_bm
#
#            if mask_thresh is not None or mask_percentile is not None:
#                array_bm, meta_bm = self.my_segment.tools.mask_array(
#                        array_bm, metadata, thresh=mask_thresh,
#                        percentile=mask_percentile, side=mask_side)
#                name_lab_dc = (name_print + '-{0}-mask-{1}-{2}.'
#                               ''.format(method, int(np.mean(wl1)),
#                                         int(np.mean(wl2))) +
#                               self.io.defaults.interleave)
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
#                                       dtype=self.io.defaults.dtype,
#                                       force=self.io.defaults.force,
#                                       ext=self.io.defaults.ext,
#                                       interleave=self.io.defaults.interleave,
#                                       byteorder=self.io.defaults.byteorder,
#                                       metadata=datacube_md)
#                if save_spec is True:
#                    spec_md = datacube_md.copy()
#                    name_label_spec = (os.path.splitext(name_lab_dc)[0] +
#                                       '-spec-mean.spec')
#                    spec_md['label'] = name_label_spec
#                    hdr_file = os.path.join(dir_out, name_label_spec + '.hdr')
#                    self.io.write_spec(hdr_file, spec_mean, spec_std,
#                                       dtype=self.io.defaults.dtype,
#                                       force=self.io.defaults.force,
#                                       ext=self.io.defaults.ext,
#                                       interleave=self.io.defaults.interleave,
#                                       byteorder=self.io.defaults.byteorder,
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
                                        self.io.defaults.interleave))
            elif method == 'ratio':
                array_bm, metadata = self.my_segment.band_math_ratio(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}.{3}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        self.io.defaults.interleave))
            elif method == 'derivative':
                array_bm, metadata = self.my_segment.band_math_derivative(
                        wl1=wl1, wl2=wl2, wl3=wl3, b1=b1, b2=b2, b3=b3,
                        list_range=list_range, print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}-{3}.{4}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        int(np.mean(wl3)),
                                        self.io.defaults.interleave))
            elif method == 'mcari2':
                array_bm, metadata = self.my_segment.band_math_mcari2(
                        wl1=wl1, wl2=wl2, wl3=wl3, b1=b1, b2=b2, b3=b3,
                        list_range=list_range, print_out=True)
                name_label = (name_print + name_append + '-{0}-{1}-{2}-{3}.{4}'
                              ''.format(method, int(np.mean(wl1)),
                                        int(np.mean(wl2)),
                                        int(np.mean(wl3)),
                                        self.io.defaults.interleave))

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
                self._plot_histogram(array_bm, fname_fig, title=name_print,
                                     xlabel=type_bm.upper(), percentile=90,
                                     fontsize=14,
                                     color='#444444')

            metadata['label'] = name_label
            metadata['interleave'] = self.io.defaults.interleave

            self._write_datacube(dir_out, name_label, array_bm, metadata)
            if geotiff is True:
                self._write_geotiff(array_bm, fname, dir_out, name_label,
                                    metadata, self.my_segment.tools)

        fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
        if os.path.isfile(fname_stats) and self.io.defaults.force is False:
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
                          self.io.defaults.interleave)

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

            metadata['interleave'] = self.io.defaults.interleave
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
#                              self.io.defaults.interleave)
#                self._write_datacube(dir_out, name_label, array_class,
#                                     metadata)

        fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
        if os.path.isfile(fname_stats) and self.io.defaults.force is False:
            df_stats_in = pd.read_csv(fname_stats)
            df_stats = df_stats_in.append(df_stats)
        df_stats.to_csv(fname_stats, index=False)

    def _execute_spat_crop(self, fname_sheet, base_dir_out, folder_name,
                           name_append, geotiff=True, method='single',
                           gdf=None):
        '''
        Actually executes the spatial crop to keep the main function a bit
        cleaner
        '''
        if isinstance(fname_sheet, pd.DataFrame):
            df_plots = fname_sheet
        elif os.path.splitext(fname_sheet)[1] == '.csv':
            df_plots = pd.read_csv(fname_sheet)

        for idx, row in df_plots.iterrows():
            cs = self._crop_read_sheet(row)
            fname = os.path.join(cs['directory'], cs['fname'])
            print('\nSpatially cropping: {0}'.format(fname))
            name_long = cs['name_long']  # `None` if it was never set
            plot_id = cs['plot_id']
            name_short = cs['name_short']
            fname_hdr = fname + '.hdr'
            self.io.read_cube(fname_hdr, name_long=name_long,
                              name_plot=plot_id, name_short=name_short)
            cs = self._pix_to_mapunit(cs)
            self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        cs['directory'], folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()
            if method == 'single':
                array_crop, metadata = self.my_spatial_mod.crop_single(
                        cs['pix_e_ul'], cs['pix_n_ul'], cs['crop_e_pix'],
                        cs['crop_n_pix'], buf_e_pix=cs['buf_e_pix'],
                        buf_n_pix=cs['buf_n_pix'])
                metadata['interleave'] = self.io.defaults.interleave
                if row['plot_id'] is not None:
                    name_plot = '_' + str(row['plot_id'])
                else:
                    name_plot = ''
                name_label = (name_print + name_plot + name_append + '.' +
                              self.io.defaults.interleave)
                fname = os.path.join(cs['directory'], cs['fname'])
                self._write_datacube(dir_out, name_label, array_crop, metadata)
                if geotiff is True:
                    self._write_geotiff(array_crop, fname, dir_out, name_label,
                                        metadata, self.my_spatial_mod.tools)
            else:
                if method == 'many_grid':
                    df_plots = self._many_grid(cs)
                elif method == 'many_gdf':
                    df_plots = self._many_gdf(cs)

                for idx, row in df_plots.iterrows():  # actually crop the image
                    # reload spyfile to my_spatial_mod??
                    self.io.read_cube(fname_hdr, name_long=name_long,
                                      name_plot=plot_id, name_short=name_short)
                    self.my_spatial_mod.load_spyfile(self.io.spyfile)
                    crop_e_pix = cs['crop_e_pix']
                    crop_n_pix = cs['crop_n_pix']
                    if pd.isnull(crop_e_pix):
                        crop_e_pix = row['crop_e_pix']
                    if pd.isnull(crop_n_pix):
                        crop_n_pix = row['crop_n_pix']
                    array_crop, metadata = self.my_spatial_mod.crop_single(
                        row['pix_e_ul'], row['pix_n_ul'],
                        crop_e_pix=crop_e_pix,
                        crop_n_pix=crop_n_pix,
                        buf_e_pix=cs['buf_e_pix'],
                        buf_n_pix=cs['buf_n_pix'],
                        plot_id=row['plot_id'], gdf=gdf)
#                    metadata = row['metadata']
#                    array_crop = row['array_crop']
                    if row['plot_id'] is not None:
                        name_plot = '_' + str(row['plot_id'])
                    else:
                        name_plot = ''
                    name_label = (name_print + name_plot + name_append + '.' +
                                  self.io.defaults.interleave)
                    metadata['interleave'] = self.io.defaults.interleave
                    fname = os.path.join(cs['directory'], cs['fname'])
                    self._write_datacube(dir_out, name_label, array_crop,
                                         metadata)
                    if geotiff is True:
                        self._write_geotiff(array_crop, fname, dir_out,
                                            name_label, metadata,
                                            self.my_spatial_mod.tools)
                self.metadata = metadata
                self.df_plots = df_plots

    def _write_datacube(self, dir_out, name_label, array, metadata):
        '''
        Writes a datacube to file using `hsio.write_cube()`
        '''
        metadata['label'] = name_label
        hdr_file = os.path.join(dir_out, name_label + '.hdr')
        self.io.write_cube(hdr_file, array,
                           dtype=self.io.defaults.dtype,
                           force=self.io.defaults.force,
                           ext=self.io.defaults.ext,
                           interleave=self.io.defaults.interleave,
                           byteorder=self.io.defaults.byteorder,
                           metadata=metadata)

    def _write_geotiff(self, array, fname, dir_out, name_label, metadata,
                       tools):
        metadata['label'] = name_label
        msg = ('Projection and Geotransform information are required for '
               'writing the geotiff. This comes from the input filename, '
               'so please be sure the correct filename is passed to '
               '`fname`.\n')
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
                          inline=True)
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
                           dtype=self.io.defaults.dtype,
                           force=self.io.defaults.force,
                           ext=self.io.defaults.ext,
                           interleave=self.io.defaults.interleave,
                           byteorder=self.io.defaults.byteorder,
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

            metadata['interleave'] = self.io.defaults.interleave
            name_label = (name_print + name_append + '.' +
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

    def _execute_spec_combine(self, fname_list, base_dir_out):
        '''
        Actually executes the spectra combine to keep the main function a bit
        cleaner
        '''
        df_specs = None
        if base_dir_out is None:
            base_dir_out = os.path.dirname(fname_list[0])
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
                           dtype=self.io.defaults.dtype,
                           force=self.io.defaults.force,
                           ext=self.io.defaults.ext,
                           interleave=self.io.defaults.interleave,
                           byteorder=self.io.defaults.byteorder,
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

            metadata['interleave'] = self.io.defaults.interleave
            name_label = (name_print + name_append + '.' +
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
            fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
            if os.path.isfile(fname_stats) and self.io.defaults.force is False:
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
        not be `None`; compares to `self.io.name_short`).

        Parameters:
            dir_search: directory to search
        '''
        msg = ('Please load a SpyFile prior to using this function')
        assert self.io.spyfile is not None, msg
        fname_kmeans = self._get_fname_similar(
                self.io.name_short, dir_search,
                search_ext=self.io.defaults.interleave, level=0)
        fpath_kmeans = os.path.join(dir_search, fname_kmeans)
        io_mask = hsio()
        io_mask.read_cube(fpath_kmeans)
        array = io_mask.spyfile.load()
        metadata = io_mask.spyfile.metadata
        return array, metadata

    def _get_class_mask(self, row, filter_cols, n_classes=1):
        '''
        Finds the class with the lowest NDVI in `row` and returns the class ID
        to be used to dictate which pixels get masked

        Parameters:
            n_classes (`int`): number of classes to mask; if 1, then will mask
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
                files that have `search_exp` in their name.
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
            base_dir_out (`str`): Parent directory that all processed datacubes
                will be saved.
            folder_name (`str`): Folder to add to `base_dir_out` to save all
                the processed datacubes.
            name_append (`str`):
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

    def _plot_histogram(self, array, fname_fig, title=None, xlabel=None,
                        percentile=90, fontsize=16, color='#444444'):
        '''
        Plots a histogram with the percentile value labeled
        '''
        if isinstance(array, np.ma.core.MaskedArray):
            array_m = array.compressed()  # allows for accurate percentile calc
        else:
            array_m = np.ma.masked_array(array, mask=False)
            array_m = array_m.compressed()

        pctl = np.nanpercentile(array_m.flatten(), percentile)

        fig, ax = plt.subplots()
        ax = sns.distplot(array_m.flatten(), bins=50, color='grey')
        data_x, data_y = ax.lines[0].get_data()

        y_lim = ax.get_ylim()
        yi = np.interp(pctl, data_x, data_y)
        ymax = yi/y_lim[1]
        ax.axvline(pctl, ymax=ymax, linestyle='--', color=color, linewidth=0.5)
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'


        legend_str = ('Percentile ({0}): {1:.3f}'
                      ''.format(percentile, pctl))
        ax.annotate(
            legend_str,
            xy=(pctl, yi),
            xytext=(0.97, 0.94),  # loc to place text
            textcoords='axes fraction',  # placed relative to axes
            ha='right',  # alignment of text
            va='top',
            fontsize=int(fontsize * 0.9),
            color=color,
            bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                      ec=(0.5, 0.5, 0.5), alpha=0.5),
            arrowprops=dict(arrowstyle='-|>',
                            color=color,
        #                    patchB=el,
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle='arc3,rad=-0.3',
                            linestyle='--',
                            linewidth=0.7))
        ax.set_title(title, fontweight='bold', fontsize=int(fontsize * 1.1))
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel('Frequency (%)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        plt.tight_layout()
        fig.savefig(fname_fig, dpi=300)

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
        same way.

        Parameters:
            method (`str`): Must be one of "ndi" (normalized difference index),
                "ratio" (simple ratio index), "derivative" (deriviative-type
                index), or "mcari2" (modified chlorophyll absorption index2).
                Indicates what kind of band
                math should be performed on the input datacube. The "ndi"
                method leverages `segment.band_math_ndi()`, the "ratio"
                method leverages `segment.band_math_ratio()`, and the
                "derivative" method leverages `segment.band_math_derivative()`.
                Please see the `segment` documentation for more information
                (default: "ndi").
            wl1 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                band math index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            wl2 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                band math index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            b1 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the first parameter of the band math index;
                if `list`, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: `None`).
            b2 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the second parameter of the band math
                index; if `list`, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: `None`).
            list_range (`bool`): Whether bands/wavelengths passed as a list is
                interpreted as a range of bands (`True`) or for each individual
                band in the list (`False`). If `list_range` is `True`,
                `b1`/`wl1` and `b2`/`wl2` should be lists with two items, and
                all bands/wavelegths between the two values will be used
                (default: `True`).
            plot_out (`bool`): whether to save a histogram of the band math
                result (default: `True`).
            geotiff (`bool`): whether to save the masked RGB image as a geotiff
                alongside the masked datacube.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
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
        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append,
                                               append_extra)
        self._execute_band_math(fname_list, base_dir_out, folder_name,
                                name_append, geotiff, method, wl1, wl2,
                                wl3, b1, b2, b3, list_range, plot_out)



#        if fname_list is not None:
#            self._execute_band_math(fname_list, base_dir_out, folder_name,
#                                    name_append, geotiff, method, wl1, wl2,
#                                    wl3, b1, b2, b3, list_range, plot_out)
#        elif base_dir is not None:
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_band_math(fname_list, base_dir_out, folder_name,
#                                    name_append, geotiff, method, wl1, wl2,
#                                    wl3, b1, b2, b3, list_range, plot_out)
#        else:  # fname_list and base_dir are both `None`
#            # base_dir may have been stored to the `batch` object
#            base_dir = self.base_dir
#            msg = ('Please set `fname_list` or `base_dir` to indicate which '
#                   'datacubes should be processed.\n')
#            assert base_dir is not None, msg
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_band_math(fname_list, base_dir_out, folder_name,
#                                    name_append, geotiff, method, wl1, wl2,
#                                    wl3, b1, b2, b3, list_range, plot_out)


    def segment_kmeans(self, fname_list=None, base_dir=None, search_ext='bip',
                       dir_level=0, base_dir_out=None, folder_name='kmeans',
                       name_append='kmeans', geotiff=True,
                       n_classes=3, max_iter=100, plot_out=True,
                       out_dtype=False, out_force=None,
                       out_ext=False, out_interleave=False,
                       out_byteorder=False):
        '''
        Batch processing tool to perform kmeans clustering on multiple
        datacubes in the same way (uses Spectral Python kmeans tool).

        Parameters:
            n_classes (`int`): number of classes to cluster (default: 3).
            max_iter (`int`): maximum iterations before terminating process
                (default: 100).
            plot_out (`bool`): whether to save a line plot of the spectra for
                each class (default: `True`).
            geotiff (`bool`): whether to save the masked RGB image as a geotiff
                alongside the masked datacube.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
        self._execute_kmeans(fname_list, base_dir_out, folder_name,
                             name_append, geotiff, n_classes, max_iter,
                             plot_out)

#        if fname_list is not None:
#            self._execute_kmeans(fname_list, base_dir_out, folder_name,
#                                 name_append, geotiff, n_classes, max_iter,
#                                 plot_out, mask_soil=False)
#        elif base_dir is not None:
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_kmeans(fname_list, base_dir_out, folder_name,
#                                 name_append, geotiff, n_classes, max_iter,
#                                 plot_out, mask_soil=False)
#        else:  # fname_list and base_dir are both `None`
#            # base_dir may have been stored to the `batch` object
#            base_dir = self.base_dir
#            msg = ('Please set `fname_list` or `base_dir` to indicate which '
#                   'datacubes should be processed.\n')
#            assert base_dir is not None, msg
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_kmeans(fname_list, base_dir_out, folder_name,
#                                 name_append, geotiff, n_classes, max_iter,
#                                 plot_out, mask_soil=False)

    def combine_kmeans_bandmath(self, fname_sheet, base_dir_out=None,
                                folder_name='mask_combine',
                                name_append='mask-combine',
                                geotiff=True, kmeans_mask_classes=1,
                                kmeans_filter='mcari2',
                                mask_percentile=90, mask_side='lower',
                                out_dtype=False, out_force=None, out_ext=False,
                                out_interleave=False, out_byteorder=False):
        '''

        Parameters:
            fname_sheet (`fname` or `Pandas.DataFrame): The filename of the
                spreadsheed that provides the necessary information for batch
                process cropping. See below for more information about the
                required and optional contents of `fname_sheet` and how to
                properly format it. Optionally, `fname_sheet` can be a
                `Pandas.DataFrame`
            kmeans_mask_classes (`int`): number of K-means classes to mask from
                the datacube. By default, the classes with the lowest average
                spectral value (e.g., NDVI, GNDVI, MCARI2, etc.; based on
                `kmeans_filter` parameter) will be masked (default: 1).
            kmeans_filter (`str`): the spectral index to base the K-means mask
                on. Must be one of 'ndvi', 'gndvi', 'ndre', or 'mcari2'. Note
                that the K-means aglorithm does not use the in its clustering
                algorithm (default: 'mcari2').
        Mask steps:
            1. load kmeans-stats.csv
            2. for each row, load spyfile based on "fname"
            3. get class to mask based on:
                a. find class with min ndvi: min_class = np.nanmin(class_X_ndvi)
            4. mask all pixels of min_class
            5. calculate band math (or load images with band math already calculated)
            6. mask all pixels below threshold/perecentile
                a. if percentile, use pctl to determine number of pixels to keep
                b. if pctl = 90, we should keep 10% of total pixels:
                    i. find total number of pixels
                    ii. calculate what 10% is (155*46 = 7130); 7130 * .1 = 713
                    iii. of pixels that are not masked by kmeans, find 713 pixels with highest bandmath
                c. if thresh, number doesn't matter
            7. apply the mask to the datacube and save spectra
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)

        if isinstance(fname_sheet, pd.DataFrame):
            df_kmeans = fname_sheet.copy()
            fname_sheet = 'dataframe passed'
        elif os.path.splitext(fname_sheet)[1] == '.csv':
            df_kmeans = pd.read_csv(fname_sheet)

        msg = ('<kmeans_filter> must be one of "ndvi", "gndvi", "ndre", '
               'or "mcari2".')
        assert kmeans_filter in ['ndvi', 'gndvi', 'ndre', 'mcari2'], msg
        filter_str = '_{0}'.format(kmeans_filter)
        filter_cols = [col for col in df_kmeans.columns if filter_str in col]

        bandmath_pctl_str = '{0}_pctl'.format(kmeans_filter)
        bandmath_side_str = '{0}_side'.format(kmeans_filter)
        columns = ['fname', 'kmeans_class', 'kmeans_nonmasked_pct',
                   bandmath_pctl_str, bandmath_side_str, 'total_nonmasked_pct']
        df_stats = pd.DataFrame(columns=columns)

        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
            fname_list = df_kmeans['fname'].tolist()
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
            df_kmeans = df_kmeans[df_kmeans['plot_id'].isin(fname_list)]

        for idx, row in df_kmeans.iterrows():  # using stats-kmeans.csv
            class_mask = self._get_class_mask(row, filter_cols,
                                              n_classes=kmeans_mask_classes)
            fname = row['fname']
            self.io.read_cube(fname)
            if base_dir_out is None:
                dir_out, name_append = self._save_file_setup(
                        os.path.dirname(fname), folder_name, name_append)
            else:
                dir_out, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            name_print = self._get_name_print()

            dir_search = os.path.join(self.io.base_dir, 'kmeans')
            array_kmeans, metadata_kmeans = self._get_array_similar(dir_search)
            dir_search = os.path.join(self.io.base_dir, 'band_math')
            array_bandmath, metadata_bandmath = self._get_array_similar(
                    dir_search)

            array_kmeans, metadata_kmeans = self.io.tools.mask_array(
                    array_kmeans, metadata_kmeans, thresh=class_mask,
                    side=None)  # when side=None, masks the exact match
            kmeans_pct = (100 * (array_kmeans.count() /
                            (array_kmeans.shape[0]*array_kmeans.shape[1])))

            # by adding the kmeans mask, hstools.mask_array will consider that
            # mask when masking by bandmath values (applicable for percentile)
            array_bandmath = np.ma.masked_array(
                    array_bandmath, mask=array_kmeans.mask)
            mask_combined, metadata_bandmath = self.io.tools.mask_array(
                    array_bandmath, metadata_bandmath,
                    percentile=mask_percentile, side=mask_side)
            total_pct = (100 * (mask_combined.count() /
                            (mask_combined.shape[0]*mask_combined.shape[1])))
            spec_mean, spec_std, datacube_masked = self.io.tools.mean_datacube(
                    self.io.spyfile, mask_combined)

            data = [os.path.basename(fname), class_mask, kmeans_pct,
                    mask_percentile, mask_side, total_pct]
            df_stats_temp = pd.DataFrame(data=[data], columns=columns)
            df_stats = df_stats.append(df_stats_temp)
            name_label = (name_print + name_append + '.' +
                          self.io.defaults.interleave)
            metadata = self.io.spyfile.metadata.copy()
            # because this is specialized, we should make our own history str
            hist_str = (" -> hs_process.batch.combine_kmeans_bandmath[<"
                        "label: 'fname_sheet?' value:{0}; "
                        "label: 'kmeans_class?' value:{1}; "
                        "label: 'mask_percentile?' value:{2}; "
                        "label: 'mask_side?' value:{3}>]"
                        "".format(fname_sheet, class_mask, mask_percentile,
                                  mask_side))
            metadata['history'] += hist_str
            self._write_datacube(dir_out, name_label, datacube_masked,
                                 metadata)

            name_label_spec = (os.path.splitext(name_label)[0] +
                               '-spec-mean.spec')
            self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
                             metadata)
        fname_stats = os.path.join(dir_out, name_append[1:] + '-stats.csv')
        if os.path.isfile(fname_stats) and self.io.defaults.force is False:
            df_stats_in = pd.read_csv(fname_stats)
            df_stats = df_stats_in.append(df_stats)
        df_stats.to_csv(fname_stats, index=False)

    def cube_to_spectra(self, fname_list=None, base_dir=None, search_ext='bip',
                        dir_level=0, base_dir_out=None,
                        folder_name='cube_to_spec',
                        name_append='cube-to-spec',
                        geotiff=True, out_dtype=False, out_force=None,
                        out_ext=False, out_interleave=False,
                        out_byteorder=False):
        '''
        Calculates the mean and standard deviation for each cube in
        `fname_sheet` and writes the result to a .spec file.

        Parameters:

        '''
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
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
                          self.io.defaults.interleave)
            metadata = self.io.spyfile.metadata.copy()
            # because this is specialized, we should make our own history str
            hist_str = (" -> hs_process.batch.cube_to_spectra[<>]")
            metadata['history'] += hist_str
            name_label_spec = (os.path.splitext(name_label)[0] +
                               '-mean.spec')
            if geotiff is True:
                self._write_geotiff(array, fname, dir_out, name_label,
                                    metadata, self.io.tools)
            # Now write spec (will change map info on metadata)
            self._write_spec(dir_out, name_label_spec, spec_mean, spec_std,
                             metadata)

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

        Parameters:
            mask_thresh (`float` or `int`): The value for which to mask the
                array; should be used with `side` parameter (default: `None`).
            mask_percentile (`float` or `int`): The percentile of pixels to
                mask; if `percentile`=95 and `side`='lower', the lowest 95% of
                pixels will be masked following the band math operation
                (default: `None`; range: 0-100).
            mask_side (`str`): The side of the threshold or percentile for
                which to apply the mask. Must be either 'lower' or 'upper'; if
                'lower', everything below the threshold/percentile will be
                masked (default: 'lower').
            geotiff (`bool`): whether to save the masked RGB image as a geotiff
                alongside the masked datacube.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
        self._execute_mask(fname_list, mask_dir, base_dir_out, folder_name,
                           name_append, geotiff, mask_thresh,
                           mask_percentile, mask_side)

#        if fname_list is not None:
#            self._execute_mask(fname_list, base_dir_out, folder_name,
#                               name_append, geotiff, mask_thresh,
#                               mask_percentile, mask_side)
#        elif base_dir is not None:
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_mask(fname_list, base_dir_out, folder_name,
#                               name_append, geotiff, mask_thresh,
#                               mask_percentile, mask_side)
#        else:  # fname_list and base_dir are both `None`
#            # base_dir may have been stored to the `batch` object
#            base_dir = self.base_dir
#            msg = ('Please set `fname_list` or `base_dir` to indicate which '
#                   'datacubes should be processed.\n')
#            assert base_dir is not None, msg
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_mask(fname_list, base_dir_out, folder_name,
#                               name_append, geotiff, mask_thresh,
#                               mask_percentile, mask_side)

    def spatial_crop(self, fname_sheet, base_dir_out=None,
                     folder_name='spatial_crop', name_append='spatial-crop',
                     geotiff=True, method='single', gdf=None, out_dtype=False,
                     out_force=None, out_ext=False, out_interleave=False,
                     out_byteorder=False):
        '''
        Iterates through spreadsheet that provides necessary information about
        how each image should be cropped and how it should be saved.

        If `gdf` is passed (a geopandas.GoeDataFrame polygon file), the cropped
        images will be shifted to the center of appropriate "plot" polygon.

        Parameters:
            fname_sheet (`fname` or `Pandas.DataFrame): The filename of the
                spreadsheed that provides the necessary information for batch
                process cropping. See below for more information about the
                required and optional contents of `fname_sheet` and how to
                properly format it. Optionally, `fname_sheet` can be a
                `Pandas.DataFrame`
            base_dir_out (`str`): output directory of the cropped image
                (default: `None`).
            folder_name (`str`): folder to add to `base_dir_out` to save all
                the processed datacubes (default: 'spatial_crop').
            name_append (`str`): name to append to the filename (default:
                'spatial-crop').
            geotiff (`bool`): whether to save an RGB image as a geotiff
                alongside the cropped datacube.
            method (`str`): Must be one of "single", "many_grid", or
                "many_gdf". Indicates whether a single plot should be cropped
                from the input datacube or if many/multiple plots should be
                cropped from the input datacube. The "single" method leverages
                `spatial_mod.crop_single()`, the "many_grid" method leverages
                `spatial_mod.crop_many_grid()`, and the "many_gdf" method
                leverages `spatial_mod.crop_many_gdf()` (there are two methods
                available to peform cropping for many/mulitple plots). Please
                see the `spatial_mod` documentation for more information
                (default: "single").
            gdf (`geopandas.GeoDataFrame`): the plot names and polygon
                geometery of each of the plots; 'plot' must be used as a column
                name to identify each of the plots, and should be an integer.
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in `batch.io.defaults, and are
                therefore accessible at a high level. See
                `hsio.set_io_defaults()` for more information on each of the
                settings.

        Note:
            `fname_sheet` may have the following required column headings:
                i) "directory", ii) "name_short", iii) "name_long", iv) "ext",
                v) "pix_e_ul", and vi) "pix_n_ul".
            With this minimum input, `batch.spatial_crop` will read in each
                image, crop from the upper left pixel (determined as
                `pix_e_ul`/`pix_n_ul`) to the lower right pixel calculated
                based on `crop_e_pix`/`crop_n_pix` (which is the width of the
                cropped area in units of pixels). `crop_e_pix` and `crop_n_pix`
                have default values, but they can also be set in `fname_sheet`,
                which will take precedence over the defaults.
            `fname_sheet` may also have the following optional column headings:
                vii) "crop_e_pix", viii) "crop_n_pix", ix) "crop_e_m",
                x) "crop_n_m", xi) "buf_e_pix", xii) "buf_n_pix",
                xiii) "buf_e_m", xiv) "buf_n_m", and xv) "plot_id".
            These optional inputs allow more control over exactly how the image
                will be cropped, and hopefully are self-explanatory until
                adequate documentation is written. Any other columns can
                be added to `fname_sheet`, but `batch.spatial_crop` does not
                use them in any way.
        '''
        if method == 'many_gdf':
            msg1 = ('Please pass a valid `geopandas.GeoDataFrame` if using '
                    'the "many_gdf" method.\n')
            msg2 = ('Please be sure the passed `geopandas.GeoDataFrame` has a '
                    'column by the name of "plot", indicating the plot ID for '
                    'each polygon geometry if using the "many_gdf" method.\n')
            assert isinstance(gdf, gpd.GeoDataFrame), msg1
            assert 'plot' in gdf.columns, msg2
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        self._execute_spat_crop(fname_sheet, base_dir_out, folder_name,
                                name_append, geotiff, method, gdf)

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
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
        self._execute_spec_clip(fname_list, base_dir_out, folder_name,
                                    name_append, wl_bands)

#        if fname_list is not None:
#            self._execute_spec_clip(fname_list, base_dir_out, folder_name,
#                                    name_append, wl_bands)
#        elif base_dir is not None:
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_spec_clip(fname_list, base_dir_out, folder_name,
#                                    name_append, wl_bands)
#        else:  # fname_list and base_dir are both `None`
#            # base_dir may have been stored to the `batch` object
#            base_dir = self.base_dir
#            msg = ('Please set `fname_list` or `base_dir` to indicate which '
#                   'datacubes should be processed.\n')
#            assert base_dir is not None, msg
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            self._execute_spec_clip(fname_list, base_dir_out, folder_name,
#                                    name_append, wl_bands)

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
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        if self.io.defaults.force is False:  # otherwise just overwrites if it exists
            fname_list = self._check_processed(fname_list, base_dir_out,
                                               folder_name, name_append)
        df_stats = self._execute_spec_smooth(
                fname_list, base_dir_out, folder_name, name_append,
                window_size, order, stats)

#        if fname_list is not None:
#            df_stats = self._execute_spec_smooth(
#                    fname_list, base_dir_out, folder_name, name_append,
#                    window_size, order, stats)
#        elif base_dir is not None:
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            df_stats = self._execute_spec_smooth(
#                    fname_list, base_dir_out, folder_name, name_append,
#                    window_size, order, stats)
#        else:  # fname_list and base_dir are both `None`
#            # base_dir may have been stored to the `batch` object
#            base_dir = self.base_dir
#            msg = ('Please set `fname_list` or `base_dir` to indicate which '
#                   'datacubes should be processed.\n')
#            assert base_dir is not None, msg
#            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
#            df_stats = self._execute_spec_smooth(
#                    fname_list, base_dir_out, folder_name, name_append,
#                    window_size, order, stats)
        if df_stats is not None:
            return df_stats

    def spectra_combine(self, fname_list=None, base_dir=None,
                        search_ext='bip', dir_level=0, base_dir_out=None,
                        out_dtype=False, out_force=None, out_ext=False,
                        out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to gather all pixels from every image in a
        directory, compute the mean and standard deviation, and save as a
        single spectra (i.e., equivalent to a single spectral pixel with no
        spatial information).

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
            out_XXX: Settings for saving the output files can be adjusted here
                if desired. They are stored in `batch.io.defaults, and are
                therefore accessible at a high level. See
                `hsio.set_io_defaults()` for more information on each of the
                settings.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        self._execute_spec_combine(fname_list, base_dir_out)

    def spectra_to_df(self, fname_list=None, base_dir=None, search_ext='spec',
                      dir_level=0):
        '''
        Reads all the .spec files in a direcory and returns their data as a
        pandas.DataFrame object.
        '''
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

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
        return df_spec
#        if isinstance(df, pd.DataFrame):
#            df_data = df.copy()
#        elif isinstance(df, str):
#            df_data = pd.read_csv(df)
#        df_join = pd.merge(df_spec, df_data, on=[join_field])
#        return df_spec

    def spectra_to_csv(self, fname_list=None, base_dir=None, search_ext='spec',
                       dir_level=0, base_dir_out=None):
        '''
        Reads all the .spec files in a direcory and saves their reflectance
        information to a .csv
        '''
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

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
        fname_csv = os.path.join(base_dir, 'stats-spectra.csv')
        df_spec.to_csv(fname_csv, index=False)
