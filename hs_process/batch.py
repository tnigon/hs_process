# -*- coding: utf-8 -*-
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import spectral.io.spyfile as SpyFile

from hs_process.utilities import defaults
from hs_process.utilities import hsio
from hs_process.utilities import hstools
from hs_process.segment import segment
from hs_process.spec_mod import spec_mod
from hs_process.spatial_mod import spatial_mod
#from hs_process import Segment


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
        if cs['crop_e_pix'] is None and cs['crop_e_m'] is not None:
            cs['crop_e_pix'] = int(cs['crop_e_m'] / spy_ps_e)
        elif cs['crop_e_pix'] is not None and cs['crop_e_m'] is None:
            cs['crop_e_m'] = cs['crop_e_pix'] * spy_ps_e
        if cs['crop_n_pix'] is None and cs['crop_n_m'] is not None:
            cs['crop_n_pix'] = int(cs['crop_n_m'] / spy_ps_n)
        elif cs['crop_n_pix'] is not None and cs['crop_n_m'] is None:
            cs['crop_n_m'] = cs['crop_n_pix'] * spy_ps_n
        # Buffer
        if cs['buf_e_pix'] is None and cs['buf_e_m'] is not None:
            cs['buf_e_pix'] = int(cs['buf_e_m'] / spy_ps_e)
        elif cs['buf_e_pix'] is not None and cs['buf_e_m'] is None:
            cs['buf_e_m'] = cs['buf_e_pix'] * spy_ps_e
        if cs['buf_n_pix'] is None and cs['buf_n_m'] is not None:
            cs['buf_n_pix'] = int(cs['buf_n_m'] / spy_ps_e)
        elif cs['buf_n_pix'] is not None and cs['buf_n_m'] is None:
            cs['buf_n_m'] = cs['buf_n_pix'] * spy_ps_e
        # Alley size
        if cs['alley_size_e_pix'] is None and cs['alley_size_e_m'] is not None:
            cs['alley_size_e_pix'] = int(cs['alley_size_e_m'] / spy_ps_e)
        elif cs['alley_size_e_pix'] is not None and cs['alley_size_e_m'] is None:
            cs['alley_size_e_m'] = cs['alley_size_e_pix'] * spy_ps_e
        if cs['alley_size_n_pix'] is None and cs['alley_size_n_m'] is not None:
            cs['alley_size_n_pix'] = int(cs['alley_size_n_m'] / spy_ps_n)
        elif cs['alley_size_n_pix'] is not None and cs['alley_size_n_m'] is None:
            cs['alley_size_n_m'] = cs['alley_size_n_pix'] * spy_ps_n
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
        msg = ('`method` must be one of either "ndi" or "ratio".\n')
        assert method in ['ndi', 'ratio'], msg

        base_dir = os.path.dirname(fname)
        if base_dir_out is None:
            dir_out, name_print, name_append = self._save_file_setup(
                    base_dir, folder_name, name_append)
        else:
            dir_out, name_print, name_append = self._save_file_setup(
                    base_dir_out, folder_name, name_append)
        if method == 'ndi':
            print('\nCalculating normalized difference index for: {0}'
                  ''.format(name_print))
        elif method == 'ratio':
            print('\nCalculating simple ratio index for: {0}'
                  ''.format(name_print))
        return dir_out, name_print, name_append

    def _execute_band_math(self, fname_list, base_dir_out, folder_name,
                           name_append, geotiff, method, wl1, wl2, b1, b2,
                           list_range, mask_thresh, mask_percentile,
                           mask_side, save_spec, save_datacube):
        '''
        Actually executes the band math to keep the main function a bit
        cleaner
        '''
        for fname in fname_list:
            self.io.read_cube(fname)
            metadata = self.io.spyfile.metadata
            dir_out, name_print, name_append = self._band_math_setup(
                    base_dir_out, folder_name, fname, name_append, method)
            self.my_segment = segment(self.io.spyfile)

            if method == 'ndi':
                array_bm, meta_bm = self.my_segment.band_math_ndi(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=True)
            elif method == 'ratio':
                array_bm, meta_bm = self.my_segment.band_math_ratio(
                        wl1=wl1, wl2=wl2, b1=b1, b2=b2, list_range=list_range,
                        print_out=True)
            meta_bm['interleave'] = self.io.defaults.interleave
            name_label_bm = (name_print + name_append + '-{0}-{1}-{2}.'
                             ''.format(method, int(np.mean(wl1)),
                                       int(np.mean(wl2))) +
                             self.io.defaults.interleave)
            meta_bm['label'] = name_label_bm

            if mask_thresh is not None or mask_percentile is not None:
                array_bm, meta_bm = self.my_segment.tools.mask_array(
                        array_bm, metadata, thresh=mask_thresh,
                        percentile=mask_percentile, side=mask_side)
                name_lab_dc = (name_print + '-{0}-mask-{1}-{2}.'
                               ''.format(method, int(np.mean(wl1)),
                                         int(np.mean(wl2))) +
                               self.io.defaults.interleave)
            # should we make an option to save a mean spectra as well?
            # Yes - we aren't required to save intermediate results and do
            # another batch process..? we get everything done in one shot -
            # after all, why do we want to do band math if we aren't also
            # calculating the average of the area (unless cropping hasn't
            # been perfomed yet)?
            # No - Keep it simpler and keep batch functions more specific in
            # their capabilities (e.g., batch.band_math, batch.mask_array,
            # batch.veg_spectra)

            if np.ma.is_masked(array_bm):
                # don't pass thresh, etc. because array is already masked
                # pass the spyfile for the metadata (tainted from threshold)
                self.io.read_cube(fname)  # read again to get fresh metadata
                self.io.spyfile.metadata['history'] = meta_bm['history']
                spec_mean, spec_std, datacube_masked, datacube_md =\
                    self.my_segment.veg_spectra(
                            array_bm, spyfile=self.io.spyfile)
                if save_datacube is True:
                    hdr_file = os.path.join(dir_out, name_lab_dc + '.hdr')
                    self.io.write_cube(hdr_file, datacube_masked,
                                       dtype=self.io.defaults.dtype,
                                       force=self.io.defaults.force,
                                       ext=self.io.defaults.ext,
                                       interleave=self.io.defaults.interleave,
                                       byteorder=self.io.defaults.byteorder,
                                       metadata=datacube_md)
                if save_spec is True:
                    spec_md = datacube_md.copy()
                    name_label_spec = (os.path.splitext(name_lab_dc)[0] +
                                       '-spec-mean.spec')
                    spec_md['label'] = name_label_spec
                    hdr_file = os.path.join(dir_out, name_label_spec + '.hdr')
                    self.io.write_spec(hdr_file, spec_mean, spec_std,
                                       dtype=self.io.defaults.dtype,
                                       force=self.io.defaults.force,
                                       ext=self.io.defaults.ext,
                                       interleave=self.io.defaults.interleave,
                                       byteorder=self.io.defaults.byteorder,
                                       metadata=spec_md)
            self._write_to_file(dir_out, name_label_bm, array_bm,
                                meta_bm, geotiff, fname, method='segment')

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
            print('Spatially cropping: {0}\n'.format(fname))
            name_long = cs['name_long']  # `None` if it was never set
            plot_id = cs['plot_id']
            name_short = cs['name_short']
            fname_hdr = fname + '.hdr'
            self.io.read_cube(fname_hdr, name_long=name_long,
                              name_plot=plot_id, name_short=name_short)
            cs = self._pix_to_mapunit(cs)
            self.my_spatial_mod = spatial_mod(self.io.spyfile, gdf)
            if base_dir_out is None:
                dir_out, name_print, name_append = self._save_file_setup(
                        cs['directory'], folder_name, name_append)
            else:
                dir_out, name_print, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            if method == 'single':
                array_crop, metadata = self.my_spatial_mod.crop_single(
                        cs['pix_e_ul'], cs['pix_n_ul'], cs['crop_e_pix'],
                        cs['crop_n_pix'])
                metadata['interleave'] = self.io.defaults.interleave
                if row['plot_id'] is not None:
                    name_plot = '-' + str(row['plot_id'])
                else:
                    name_plot = ''
                name_label = (name_print + name_append + name_plot + '.' +
                              self.io.defaults.interleave)
                metadata['label'] = name_label
                fname = os.path.join(cs['directory'], cs['fname'])
                self._write_to_file(dir_out, name_label, array_crop,
                                         metadata, geotiff, fname, method='spatial')
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
                    array_crop, metadata = self.my_spatial_mod.crop_single(
                        row['pix_e_ul'], row['pix_n_ul'],
                        crop_e_pix=cs['crop_e_pix'],
                        crop_n_pix=cs['crop_n_pix'],
                        buf_e_pix=cs['buf_e_pix'],
                        buf_n_pix=cs['buf_n_pix'],
                        plot_id=row['plot_id'], gdf=gdf)
#                    metadata = row['metadata']
#                    array_crop = row['array_crop']
                    metadata['interleave'] = self.io.defaults.interleave
                    if row['plot_id'] is not None:
                        name_plot = '-' + str(row['plot_id'])
                    else:
                        name_plot = ''
                    name_label = (name_print + name_append + name_plot + '.' +
                                  self.io.defaults.interleave)
                    metadata['label'] = name_label
                    fname = os.path.join(cs['directory'], cs['fname'])
                    self._write_to_file(dir_out, name_label, array_crop,
                                             metadata, geotiff, fname, method='spatial')
                self.metadata = metadata
                self.df_plots = df_plots

    def _write_to_file(self, dir_out, name_label, array, metadata,
                            geotiff=False, fname=None, method='spatial'):
        '''
        method (`str`): must be either "spatial", "segment", or "spectral".
            Determines which object to get "map info" from.
        '''

        hdr_file = os.path.join(dir_out, name_label + '.hdr')
        self.io.write_cube(hdr_file, array,
                           dtype=self.io.defaults.dtype,
                           force=self.io.defaults.force,
                           ext=self.io.defaults.ext,
                           interleave=self.io.defaults.interleave,
                           byteorder=self.io.defaults.byteorder,
                           metadata=metadata)
        if geotiff is True:
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

            map_info_set = metadata['map info']
            if method == 'spatial':
                ul_x_utm = self.my_spatial_mod.tools.get_meta_set(map_info_set, 3)
                ul_y_utm = self.my_spatial_mod.tools.get_meta_set(map_info_set, 4)
                size_x_m = self.my_spatial_mod.tools.get_meta_set(map_info_set, 5)
                size_y_m = self.my_spatial_mod.tools.get_meta_set(map_info_set, 6)
            if method == 'segment':
                ul_x_utm = self.my_segment.tools.get_meta_set(map_info_set, 3)
                ul_y_utm = self.my_segment.tools.get_meta_set(map_info_set, 4)
                size_x_m = self.my_segment.tools.get_meta_set(map_info_set, 5)
                size_y_m = self.my_segment.tools.get_meta_set(map_info_set, 6)
            if method == 'spectral':
                ul_x_utm = self.my_spectral_mod.tools.get_meta_set(map_info_set, 3)
                ul_y_utm = self.my_spectral_mod.tools.get_meta_set(map_info_set, 4)
                size_x_m = self.my_spectral_mod.tools.get_meta_set(map_info_set, 5)
                size_y_m = self.my_spectral_mod.tools.get_meta_set(map_info_set, 6)

            # Note the last pixel size must be negative to begin at upper left
            geotransform_out = [ul_x_utm, size_x_m, 0.0, ul_y_utm, 0.0,
                                -size_y_m]
            self.io.write_tif(fname_tif, spyfile=array,
                              projection_out=projection_out,
                              geotransform_out=geotransform_out)

    def _execute_spec_clip(self, fname_list, base_dir_out, folder_name,
                           name_append, wl_bands):
        '''
        Actually executes the spectral clip to keep the main function a bit
        cleaner
        '''
        for fname in fname_list:
            print('\nSpectrally clipping: {0}'.format(fname))
            self.io.read_cube(fname)  # options: name_long, name_plot, name_short, individual_plot, overwrite
            self.my_spectral_mod = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_print, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_print, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
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
            self.io.read_spec(fname)  # options: name_long, name_plot, name_short, individual_plot, overwrite
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
            df_smooth_stats = pd.DataFrame(columns=['fname', 'mean', 'std', 'cv'])

        for fname in fname_list:
            print('Spectrally smoothing: {0}\n'.format(fname))
            self.io.read_cube(fname)  # options: name_long, name_plot, name_short, individual_plot, overwrite
            self.my_spectral_mod = spec_mod(self.io.spyfile)
            base_dir = os.path.dirname(fname)
            if base_dir_out is None:
                dir_out, name_print, name_append = self._save_file_setup(
                        base_dir, folder_name, name_append)
            else:
                dir_out, name_print, name_append = self._save_file_setup(
                        base_dir_out, folder_name, name_append)
            print(name_append)
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
            if name_append[0] == '-':
                name_append = name_append[1:]
            fname_stats = os.path.join(dir_out, name_append +
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
        name_print = self.io.name_short
        if name_append is None:
            name_append = ''
        else:
            if name_append[0] != '-':
                name_append = '-' + str(name_append)
        msg = ('Could not get a name for input datacube.\n')
        assert name_print is not None, msg
        return dir_out, name_print, name_append
#        if fname is not None:
#            name_print = self.io.name_short
#        else:
#            name_print = self.name_short
#        return dir_out


    def segment_band_math(self, fname_list=None, base_dir=None,
                          search_ext='bip', dir_level=0, base_dir_out=None,
                          folder_name='band_math', name_append='band-math',
                          geotiff=True, method='ndi', wl1=None, wl2=None,
                          b1=None, b2=None, list_range=True, mask_thresh=None,
                          mask_percentile=None, mask_side='lower',
                          save_spec=True, save_datacube=True,
                          out_dtype=False, out_force=None, out_ext=False,
                          out_interleave=False, out_byteorder=False):
        '''
        Batch processing tool to perform band math on multiple datacubes in the
        same way.

        Parameters:
            method (`str`): Must be one of "ndi" (normalized difference index),
                or "ratio" (simple ratio index). Indicates what kind of band
                math should be performed on the input datacube. The "ndi"
                method leverages `segment.band_math_ndi()` and the "ratio"
                method leverages `segment.band_math_ratio()`. Please see the
                `segment` documentation for more information (default:
                "ndi").
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
            save_spec (`bool`): Whether to save a spectrum to file representing
                the mean value of all unmasked pixels in the array after band
                math (default: `True`).
            save_datacube (`bool`): Whether to save a datacube to file
                containing only the unmasked pixels after the band math
                operation; only applies if `mask_thresh` or `mask_percentile`
                are not `None` (default: `True`).
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)
        if fname_list is not None:
            self._execute_band_math(fname_list, base_dir_out, folder_name,
                                    name_append, geotiff, method, wl1, wl2,
                                    b1, b2, list_range, mask_thresh,
                                    mask_percentile, mask_side, save_spec,
                                    save_datacube)
        elif base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_band_math(fname_list, base_dir_out, folder_name,
                                    name_append, geotiff, method, wl1, wl2,
                                    b1, b2, list_range, mask_thresh,
                                    mask_percentile, mask_side, save_spec,
                                    save_datacube)
        else:  # fname_list and base_dir are both `None`
            base_dir = self.base_dir  # base_dir may have been stored to the `batch` object
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_band_math(fname_list, base_dir_out, folder_name,
                                    name_append, geotiff, method, wl1, wl2,
                                    b1, b2, list_range, mask_thresh,
                                    mask_percentile, mask_side, save_spec,
                                    save_datacube)

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
            df_stats = self._execute_spec_smooth(
                    fname_list, base_dir_out, folder_name, name_append,
                    window_size, order, stats)
        elif base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            df_stats = self._execute_spec_smooth(
                    fname_list, base_dir_out, folder_name, name_append,
                    window_size, order, stats)
        else:  # fname_list and base_dir are both `None`
            base_dir = self.base_dir  # base_dir may have been stored to the `batch` object
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            df_stats = self._execute_spec_smooth(
                    fname_list, base_dir_out, folder_name, name_append,
                    window_size, order, stats)
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
        if fname_list is not None:
            self._execute_spec_combine(fname_list, base_dir_out)
        elif base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_spec_combine(fname_list, base_dir_out)
        else:  # fname_list and base_dir are both `None`
            base_dir = self.base_dir  # base_dir may have been stored to the `batch` object
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
            self._execute_spec_combine(fname_list, base_dir_out)
