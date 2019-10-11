# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd

from hs_process.utilities import hsio
from hs_process.utilities import hstools
from hs_process import spec_mod
from hs_process import spatial_mod
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
#        print(hsio)
#        try:
#            self.io = hsio()
#        except TypeError:
#            self.io = hsio.hsio()
        self.io = hsio()

    def _execute_spat_crop(self, fname_sheet, base_dir_out, folder_name,
                           name_append, crop_e_pix, crop_n_pix, geotiff):
        '''
        Actually executes the spatial crop to keep the main function a bit
        cleaner
        '''
        df_plots = pd.read_csv(fname_sheet)

        for idx, row in df_plots.iterrows():
            directory = row['directory']
            name_short = row['name_short']
            name_long = row['name_long']
            ext = row['ext']
            fname = os.path.join(directory, name_short+name_long+ext)
            print('Spatially cropping: {0}\n'.format(fname))

            pix_e_ul = row['pix_e_ul']
            pix_n_ul = row['pix_n_ul']

            self.io.read_cube(fname)  # options: name_long, name_plot, name_short, individual_plot, overwrite
            sm = spatial_mod(self.io.spyfile)
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
            else:
                name_append = '-' + str(name_append)

            array_crop, metadata = sm.crop_single(pix_e_ul, pix_n_ul,
                                                  crop_e_pix, crop_n_pix)
            metadata['interleave'] = self.io.defaults.interleave
            name_label = (name_print + name_append + '.' +
                          self.io.defaults.interleave)
            metadata['label'] = name_label

            hdr_file = os.path.join(dir_out, name_label + '.hdr')
            self.io.write_cube(hdr_file, array_crop,
                               dtype=self.io.defaults.dtype,
                               force=self.io.defaults.force,
                               ext=self.io.defaults.ext,
                               interleave=self.io.defaults.interleave,
                               byteorder=self.io.defaults.byteorder,
                               metadata=metadata)
            if geotiff is True:
                fname_tif = os.path.join(dir_out, name_label + '.tif')
                img_ds = self.io._read_envi_gdal(fname_in=fname)
                projection_out = img_ds.GetProjection()
                geotransform_out = img_ds.GetGeotransform()
                img_ds = None  # I only want to use GDAL when I have to..

                map_info_set = metadata['map info']
                ul_x_utm = self.tools.get_meta_set(map_info_set, 3)
                ul_y_utm = self.tools.get_meta_set(map_info_set, 4)
                geotransform_out = [ul_x_utm, sm.size_x_m, 0.0, ul_y_utm, 0.0,
                                    sm.size_y_m]
                self.io.write_tif(fname_tif, spyfile=array_crop,
                                  projection_out=projection_out,
                                  geotransform_out=geotransform_out)

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
            else:
                name_append = '-' + str(name_append)

            array_clip, metadata = sm.spectral_clip(wl_bands=wl_bands)

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

        df_mean = df_specs.mean()
        df_mean = df_mean.rename('mean')
        df_std = df_specs.std()
        df_std = df_std.rename('std')
        df_cv = df_mean / df_std
        df_cv = df_cv.rename('cv')

        hdr_file = os.path.join(base_dir_out, 'spec_mean_spy.spec.hdr')
        self.io.write_spec(hdr_file, df_mean, df_std,
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
            else:
                name_append = '-' + str(name_append)

            array_smooth, metadata = sm.spectral_smooth(
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

    def spatial_crop(self, fname_sheet, crop_e_pix=90, crop_n_pix=120,
                     base_dir_out=None,
                     folder_name='spatial_crop',
                     name_append='spatial-crop', geotiff=True,
                     out_dtype=False, out_force=None, out_ext=False,
                     out_interleave=False, out_byteorder=False):
        '''
        Iterates through spreadsheet that provides necessary information about
        how each image should be cropped and how it should be saved

        Parameters:
            fname_sheet (`fname`): The filename of the spreadsheed that
                provides the necessary information for batch process cropping.
                See below for more information about the required and optional
                contents of `fname_sheet` and how to properly format it.
            crop_e_pix (`int`): number of pixels to allocate per row in the
                cropped image (default: 90).
            crop_n_pix (`int`): number of pixels per colum in the cropped image
                 (default: 120).
            base_dir_out (`str`): output directory of the cropped image
                (default: `None`).
            folder_name (`str`): folder to add to `base_dir_out` to save all
                the processed datacubes (default: 'spatial_crop').
            name_append (`str`): name to append to the filename (default:
                'spatial-crop').
            geotiff (`bool`): whether to save an RGB image as a geotiff
                alongside the cropped datacube.
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
                x) "crop_n_m", xi) "buffer_x_pix", xii) "buffer_y_pix",
                xiii) "buffer_x_m", xiv) "buffer_y_m", and xv) "plot_id".
            These optional inputs allow more control over exactly how the image
                will be cropped, and hopefully are self-explanatory until
                adequate documentation is written. Any other columns can
                be added to `fname_sheet`, but `batch.spatial_crop` does not
                use them in any way.
        '''
        self.io.set_io_defaults(out_dtype, out_force, out_ext, out_interleave,
                                out_byteorder)

        if os.path.splitext(fname_sheet)[1] == '.csv':
            self._execute_spat_crop(fname_sheet, base_dir_out, folder_name,
                                    name_append, crop_e_pix, crop_n_pix)

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
