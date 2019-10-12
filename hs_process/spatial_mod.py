# -*- coding: utf-8 -*-
import geopandas as gpd
import json
import numpy as np
import os
import pandas as pd
from shapely.geometry import Polygon
import spectral.io.spyfile as SpyFile

from hs_process.utilities import defaults
from hs_process.utilities import hstools


class spatial_mod(object):
    '''
    Class for manipulating data within the spatial domain (e.g., cropping by
    geographical boundary)
    '''
    def __init__(self, spyfile, gdf=None):
        '''
        spyfile (`SpyFile` object): The Spectral Python datacube to manipulate.
        gdf (`geopandas.DataFrame`): Polygon data that includes the plot_id and
            its geometry.
        '''
        self.spyfile = spyfile
        self.gdf = gdf
        self.tools = hstools(spyfile)

#        self.pix_e_ul = None  # these should be passed (can't use defaults)
#        self.pix_n_ul = None
        self.crop_e_pix = None  # defaults set by `batch.crop_single` function
        self.crop_n_pix = None
        self.crop_e_m = None
        self.crop_n_m = None

        self.buf_x_pix = None
        self.buf_y_pix = None
        self.buffer_x_m = None
        self.buffer_y_m = None

        self.geotransform = None
#        self.alley_size_pix = int(6.132 / -0.04)  # alley - skip 6.132 m
        self.projection = None
        self.plot_cols = None
        self.plot_rows = None
        self.row_plots_top = 0
        self.row_plots_bot = 0

        self.spy_ps_e = None
        self.spy_ps_n = None
        self.spy_ul_e_srs = None
        self.spy_ul_n_srs = None

        self.defaults = defaults
#        def run_init():
#            try:
#                self.spy_ps_e = float(self.spyfile.metadata['map info'][5])
#                self.spy_ps_n = float(self.spyfile.metadata['map info'][6])
#                self.spy_ul_e_srs = float(self.spyfile.metadata['map info'][4])
#                self.spy_ul_n_srs = float(self.spyfile.metadata['map info'][3])
#            except KeyError as e:
#                self.spy_ps_e = None
#                self.spy_ps_n = None
#                self.spy_ul_e_srs = None
#                self.spy_ul_n_srs = None
#
#        run_init()
        self.load_spyfile(spyfile)

    def _create_spyfile_extent_gdf(self, spyfile, metadata=None, epsg=32615):
        '''
        '''
        if metadata is None:
            metadata = spyfile.metadata
        crs = {'init': 'epsg:{0}'.format(epsg)}

        e_m = float(metadata['map info'][5])  # pixel size
        n_m = float(metadata['map info'][6])
        size_x = spyfile.shape[1]  # number of pixels
        size_y = spyfile.shape[0]
        srs_e_m = float(metadata['map info'][3])  # UTM coordinate
        srs_n_m = float(metadata['map info'][4])

        e_nw = srs_e_m
        e_ne = srs_e_m + (size_x * e_m)
        e_se = srs_e_m + (size_x * e_m)
        e_sw = srs_e_m
        n_nw = srs_n_m
        n_ne = srs_n_m
        n_se = srs_n_m + (size_y * n_m)
        n_sw = srs_n_m + (size_y * n_m)
        coords_e = [e_nw, e_ne, e_se, e_sw, e_nw]
        coords_n = [n_nw, n_ne, n_se, n_sw, n_nw]

        polygon_geom = Polygon(zip(coords_e, coords_n))
        gdf_sp = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
        return gdf_sp

    def _overlay_gdf(self, spyfile, gdf, epsg_sp=32615, how='intersection'):
        '''
        Performs a geopandas overlay between the input geodatafram (`gdf`) and
        the extent of `spyfile`.
        '''
        gdf_sp = self._create_spyfile_extent_gdf(spyfile, epsg=epsg_sp)
        gdf_filter = gpd.overlay(gdf, gdf_sp, how=how)
        return gdf_filter

    def _find_plots_shp(self, spyfile, gdf, plot_id_ref, pix_e_ul, pix_n_ul,
                        metadata=None):
        '''
        Calculates the number of x plots and y plots in image, determines
        the plot ID number, and calculates and records start/end pixels for
        each plot
        '''
        columns=['plot_id', 'pix_e_ul', 'pix_n_ul']
        df_plots = pd.DataFrame(columns=columns)
        gdf_filter = self._overlay_gdf(spyfile, gdf)
        msg = ('Please be sure the reference plot (`plot_id_ref`): {0} is '
               'within the spatial extent of the datacube (`spyfile`):  {1}\n'
               ''.format(plot_id_ref, spyfile.filename))
        assert plot_id_ref in gdf_filter['plot'].tolist(), msg

        if metadata is None:
            metadata = spyfile.metadata
#        spy_ps_e = float(metadata['map info'][5])  # pixel size
#        spy_ps_n = float(metadata['map info'][6])
        spy_srs_e_m = float(metadata['map info'][3])  # UTM coordinate
        spy_srs_n_m = float(metadata['map info'][4])

        for idx, row in gdf_filter.iterrows():
            plot = row['plot']
            bounds = row['geometry'].bounds
            plot_srs_e_m = bounds[0]
            plot_srs_n_m = bounds[3]
            # plot offset from datacube
            offset_e = int((plot_srs_e_m - spy_srs_e_m) / self.spy_ps_e)
            offset_n = int((plot_srs_n_m - spy_srs_n_m) / self.spy_ps_n)
            data = [plot, offset_e, offset_n]
            df_plots_temp = pd.DataFrame(columns=columns, data=[data])
            df_plots = df_plots.append(df_plots_temp, ignore_index=True)
        return df_plots

    def envi_crop(self, plot_id_ul, ul_x_pix=100, ul_y_pix=100,
                  plot_size_e_m=9.170, plot_size_n_m=3.049,
                  alley_size_n_m=6.132,
                  buf_x_m=1.0, buf_y_m=0.5, n_plots_x=5, n_plots_y=None,
                  spyfile=None):
        '''
        n_plots_x: number of plots in a row in east/west direction
        n_plots_y: number of plots in a row in north/south direction
        plot_size_e_m: dimension of each plot in the E/W direction (in map units)
        plot_size_n_m: dimension of each plot in the N/S direction (in map units)
        '''
        if spyfile is None:
            spyfile = self.spyfile
            metadata = spyfile.metadata
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            metadata = spyfile.metadata

        plot_size_e_pix = int(plot_size_e_m / self.spy_ps_e)
        plot_size_n_pix = int(plot_size_n_m / self.spy_ps_n)
        buf_x_pix = int(buf_x_m / self.spy_ps_e)
        buf_y_pix = int(buf_y_m / self.spy_ps_n)

#        self.cols_pix = ds_img.RasterXSize
#        self.rows_pix = ds_img.RasterYSize
#        self.bands = ds_img.RasterCount
#        self.geotransform = ds_img.GetGeoTransform()
#        self.spy_ul_e_srs = self.geotransform[0]
#        self.spy_ul_n_srs = self.geotransform[3]
#        self.spy_ps_e = self.geotransform[1]
#        self.spy_ps_n = self.geotransform[5]
#        self.plot_size_e_pix = int(self.plot_size_e_m / self.spy_ps_e)
#        self.plot_size_n_pix = int(self.plot_size_n_m / self.spy_ps_n)
#        self.buf_x_pix = int(self.buf_x_m / self.spy_ps_e)
#        self.buf_y_pix = int(self.buf_y_m / self.spy_ps_n)
#        n_plots_x = 5  # should always be 5
        if n_plots_y is None:
            n_plots_y = int(abs((spyfile.nrows - ul_y_pix) /
                                      plot_size_n_pix))
        alley_size_pix = int(alley_size_n_m / self.spy_ps_n)  # alley - skip 6.132 m

        self.df_shp = pd.DataFrame(columns=['plot_id', 'ul_x_utm', 'ul_y_utm'])

        row_plots_top, row_plots_bot = self._check_alley(
                plot_id_ul, n_plots_y, rows_pix, ul_y_pix,
                plot_size_n_pix, alley_size_pix)
        df_plots = self._calc_size(n_plots_x, row_plots_top, row_plots_bot,
                                   plot_size_e_pix, plot_size_n_pix, ul_x_pix,
                                   ul_y_pix)
        return df_plots

    def _record_pixels(plot_n_start, plot_n_end, row_plot, df_plots, plot_size_e_pix,
                       plot_size_n_pix, ul_x_pix, ul_y_pix):
        '''
        '''
        for plot_n in range(plot_n_start, plot_n_end):  # top plots
            col_plot = (plot_n) % 5
            if col_plot == 0:
                row_plot += 1
            plot_id = plot_id_ul - (col_plot * 100)  # E/W
            plot_id = plot_id - row_plot  # N/S
            col_pix = (col_plot * plot_size_e_pix) + ul_x_pix
            row_pix = (row_plot * plot_size_n_pix) + ul_y_pix
            df_temp = pd.DataFrame(data=[[plot_id, col_plot, row_plot,
                                          col_pix, row_pix]],
                                   columns=df_plots.columns)
            df_plots = df_plots.append(df_temp, ignore_index=True)
            return df_plots, row_plot

    def _calc_size(n_plots_x, row_plots_top, row_plots_bot, plot_size_e_pix, plot_size_n_pix, ul_x_pix, ul_y_pix):
        '''
        Calculates the number of x plots and y plots in image, determines
        the plot ID number, and calculates and records start/end pixels for
        each plot
        '''
        df_plots = pd.DataFrame(columns=['plot_id', 'col_plot',
                                      'row_plot', 'col_pix',
                                      'row_pix'])
        row_plot = -1
        plot_n_start = 0
        plot_n_end = n_plots_x * row_plots_top

        df_plots, row_plot = self._record_pixels(
                plot_n_start, plot_n_end, row_plot, df_plots, plot_size_e_pix,
                plot_size_n_pix, ul_x_pix, ul_y_pix)

        if row_plots_bot > 0:  # do the same for bottom, adjusting start/end
            plot_n_start = plot_n_end
            plot_n_bot = n_plots_x * row_plots_bot
            plot_n_end = plot_n_end + plot_n_bot
            df_plots, row_plot = self._record_pixels(
                    plot_n_start, plot_n_end, row_plot, df_plots, plot_size_e_pix,
                    plot_size_n_pix, ul_x_pix, ul_y_pix)

        return df_plots

    def _find_plots_hard(self, n_plots_x=5, n_plots_y=9, plot_size_e_m=7.6,
                         plot_size_n_m=3.048):
        '''
        '''
#        columns=['plot_id', 'pix_e_ul', 'pix_n_ul']
#        df_plots = pd.DataFrame(columns=columns)
#
#        spy_ps_e = float(metadata['map info'][5])  # pixel size
#        spy_ps_n = float(metadata['map info'][6])
#        plot_size_e_m = 7.6  # meters
#        plot_size_n_m = 3.048  # meters
#        plot_size_e_pix = int(plot_size_e_m / self.spy_ps_e)
#        plot_size_n_pix = int(plot_size_n_m / self.spy_ps_n)

    def _check_alley(plot_id_ul, n_plots_y, rows_pix,
                     ul_y_pix, plot_size_n_pix, alley_size_pix):
        '''
        Calculates whether there is an alleyway in the image (based on plot
        configuration), then adjusts n_plots_y so it is correct after
        considering the alley
        '''
        plot_id_tens = abs(plot_id_ul) % 100
        row_plots_top = plot_id_tens % n_plots_y
        if row_plots_top == 0:
            row_plots_top = n_plots_y  # remainder is 0, not 9..

        if row_plots_top < n_plots_y:
            # get pix left over
            pix_remain = (rows_pix - abs(ul_y_pix) -
                          (row_plots_top * abs(plot_size_n_pix)))
        else:
            row_plots_bot = 0
            return row_plots_top, row_plots_bot

        if pix_remain >= abs(alley_size_pix + plot_size_n_pix):
            # have room for more plots (must still remove 2 rows of plots)
            # calculate rows remain after skip
            row_plots_bot = int(abs((pix_remain + alley_size_pix) /
                                     plot_size_n_pix))
            n_plots_y = row_plots_top + row_plots_bot
        # these are for if alley_size_pix is large but plot_size_n_pix is relatively small..
        else:
            n_plots_y = row_plots_top
#        elif pix_remain >= abs(plot_size_n_pix) * 2:
#            # remove 2 rows of plots
#            n_plots_y -= 2
#        elif pix_remain >= abs(plot_size_n_pix):
#            # remove 1 row of plots
#            n_plots_y -= 1
#        else:
#            # works out perfect.. don't have to change anything
#            pass
        return row_plots_top, row_plots_bot


    def _get_corners(self, pix_ul, crop_pix, buffer_pix):
        '''
        Gets the upper left and lower right corner of the cropped array. If
        necessary, applies the buffer to the coordinates. This is a generic
        function that can be used in either the easting or northing direction.

        Parameters:
            pix_ul (`int`): upper left pixel coordinate (can be either easting
                or northing direction).
            crop_pix (`int`): number of pixels to be cropped (before applying
                the buffer).
            buffer_pix (`int`): number of pixels to buffer

        Returns:
            pix_ul (`int`): upper left pixel coordinate after applying the
                buffer
            pix_lr (`int`): lower right pixel coordinate after applying the
                buffer
        '''
        pix_lr = pix_ul + crop_pix
        if buffer_pix is not None:
            pix_ul += buffer_pix
            pix_lr -= buffer_pix
        return pix_ul, pix_lr

    def _handle_defaults(self, crop_e_pix, crop_n_pix, crop_e_m, crop_n_m):
        '''
        If these are set to `None`, retrieves default values from
        `spatial_mod.defaults`, which can be accessed and modified by an
        instance of this class by a higher level program.
        '''
        if crop_e_pix is None and crop_e_m is None:
            crop_e_pix = self.defaults.crop_defaults['crop_e_pix']
        if crop_n_pix is None and crop_n_m:
            crop_n_pix = self.defaults.crop_defaults['crop_n_pix']
        return crop_e_pix, crop_n_pix, crop_e_m, crop_n_m

    def _read_plot_shp(self):
        '''
        Reads shapefile of plot bounds and record upper left (northwest)
        corner of each plot
        '''
        assert self.df_shp is not None, 'Please load a shapefile\n'
        df_shp = self.df_shp.copy()
        drv = ogr.GetDriverByName('ESRI Shapefile')
        ds_shp = drv.Open(self.fname_shp, 0)
        if ds_shp is None:
            print('Could not open {0}'.format(self.fname_shp))
        layer = ds_shp.GetLayer()

        for feat in layer:
            geom = feat.GetGeometryRef()
            bounds = geom.GetBoundary()
            bounds_dict = json.loads(bounds.ExportToJson())
            bounds_coords = bounds_dict['coordinates']
            plot_id = feat.GetField('plot')
            x, y = zip(*bounds_coords)
            ul_x_utm = min(x)
            ul_y_utm = max(y)
            df_temp = pd.DataFrame(data=[[plot_id, ul_x_utm, ul_y_utm]],
                                   columns=df_shp.columns)
            df_shp = df_shp.append(df_temp, ignore_index=True)
            self.df_shp = df_shp

    def crop_many(self, pix_e_ul, pix_n_ul, crop_e_pix=None,
                  crop_n_pix=None, crop_e_m=None, crop_n_m=None,
                  buffer_x_pix=None, buffer_y_pix=None,
                  buffer_x_m=None, buffer_y_m=None,
                  spyfile=None, gdf=None):
        '''
        Crops many plots from a single image by comparing the image to a
        polygon file that contains plot information and geometry of plot
        boundaries.

        Stores each cropped array and its metadata in a
        '''
        df_croped = pd.DataFrame(colummns=[['array', 'metadata']])


        # store all the necessary cropping information in df_plots (e.g., pix_e_ul, pix_n_ul, etc.)
        for idx, row in df_plots.iterrows():
            array_crop, metadata = self.crop_single(
                    pix_e_ul, pix_n_ul, plot_size_e_pix=90, plot_size_n_pix=120,
                    spyfile=None)
        df_cropped_temp = pd.DataFrame(colummns=[['array', 'metadata']],
                                       data=[[array_crop, metadata]])


        if base_dir_crop is None:
            base_dir_crop = os.path.join(self.base_dir, 'crop')
        if not os.path.isdir(base_dir_crop):
            os.mkdir(base_dir_crop)
#        img_crop = self.ds_in.ReadAsArray(xoff, yoff, xsize, ysize)
        for idx, row in self.df_plots.iterrows():
            plot_id = row['plot_id']
            col_pix = row['col_pix']
            row_pix = row['row_pix']

#        col_pix = df_plots[df_plots['plot_id'] == 2025]['col_pix'].item()
#        row_pix = abs(df_plots[df_plots['plot_id'] == 2025]['row_pix'].item())
            array_img_crop = self.img_ds.ReadAsArray(
                    xoff=abs(col_pix), yoff=abs(row_pix),
                    xsize=abs(self.plot_size_e_pix - (self.buf_x_pix * 2)),
                    ysize=abs(self.plot_size_n_pix - (self.buf_y_pix * 2)))


            array_img_crop = self.img_sp.read_subregion(
                    (abs(row_pix), abs(row_pix) + abs(self.plot_size_n_pix - (self.buf_y_pix * 2))),
                    (abs(col_pix), abs(col_pix) + abs(self.plot_size_e_pix - (self.buf_x_pix * 2))))
            base_name = os.path.basename(self.fname_in)
            base_name_short = base_name[:base_name.find('gige_') + 7]  # limit of 2 digits in image number (i.e., max of 99)
            if base_name_short[-1] == '_' or base_name_short[-1] == '-':
                base_name_short = base_name_short[:-1]
            fname_out_envi = os.path.join(
                    base_dir_crop, (base_name_short + '_' + str(plot_id) +
                                    '.bsq'))
            print('Cropping plot {0}'.format(plot_id))
            utm_x = self.geotransform[0]
            utm_y = self.geotransform[3]
#            print(plot_id)
#            print(self.df_shp[self.df_shp['plot_id'] == plot_id]['ul_x_utm'])
            if self.fname_shp is not None:
                ul_x_utm = (self.df_shp[self.df_shp['plot_id'] == plot_id]
                            ['ul_x_utm'].item() + self.buf_x_m)
                ul_y_utm = (self.df_shp[self.df_shp['plot_id'] == plot_id]
                            ['ul_y_utm'].item() - self.buf_y_m)
            else:
                ul_x_utm, ul_y_utm = self._get_UTM(col_pix, row_pix, utm_x,
                                                   utm_y, size_x=self.spy_ps_e,
                                                   size_y=self.spy_ps_n)
            geotransform_out = [ul_x_utm, self.spy_ps_e, 0.0, ul_y_utm, 0.0,
                                self.spy_ps_n]
            self._write_envi(array_img_crop, fname_out_envi, geotransform_out)
            self._modify_hdr(fname_out_envi)
            fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'
            self._write_tif(array_img_crop, fname_out_tif, geotransform_out)

    def crop_single(self, pix_e_ul, pix_n_ul, crop_e_pix=None,
                    crop_n_pix=None, crop_e_m=None, crop_n_m=None,
                    buffer_x_pix=None, buffer_y_pix=None,
                    buffer_x_m=None, buffer_y_m=None,
                    spyfile=None):
        '''
        Crops a single plot from an image.

        Parameters:
            pix_e_ul (`int`): upper left column (easting)to begin cropping
            pix_n_ul (`int`): upper left row (northing) to begin cropping
            crop_e_pix (`int`, optional): number of pixels for each row in the
                cropped image
            crop_n_pix (`int`, optional): number of pixels for each column in
                the cropped image
            crop_e_m (`float`, optional): length of each row (easting
                     direction) in the cropped image (in map units; e.g.,
                     meters).
            crop_n_m (`float`, optional): length of each column (northing
                     direction) in the cropped image (in map units; e.g.,
                     meters).
            buffer_x_pix (`int`, optional): applied/calculated after
                `crop_e_pix` and `crop_n_pix` are accounted for
            buffer_y_pix (`int`, optional):
            buffer_x_m (`float`, optional):
            buffer_y_m (`float`, optional):
            spyfile (`SpyFile` object or `numpy.ndarray`): The data cube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).

        Returns:
            array_crop (`numpy.ndarray`): Cropped datacube
            metadata (`dict`): Modified metadata describing the cropped
                hyperspectral datacube (`array_crop`).
        '''
        crop_e_pix, crop_n_pix, crop_e_m, crop_n_m = self._handle_defaults(
                crop_e_pix, crop_n_pix, crop_e_m, crop_n_m)
        pix_e_ul, pix_e_lr = self._get_corners(pix_e_ul, crop_e_pix,
                                               buffer_x_pix)
        pix_n_ul, pix_n_lr = self._get_corners(pix_n_ul, crop_n_pix,
                                               buffer_y_pix)

        if spyfile is None:
            spyfile = self.spyfile
            array_crop = spyfile.read_subregion((pix_n_ul, pix_n_lr),
                                                (pix_e_ul, pix_e_lr))
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array_crop = spyfile.read_subregion((pix_n_ul, pix_n_lr),
                                                (pix_e_ul, pix_e_lr))
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile
            array_crop = array[pix_n_ul:pix_n_lr, pix_e_ul:pix_e_lr, :]

        metadata = self.tools.spyfile.metadata
        map_info_set = metadata['map info']
        utm_x = self.tools.get_meta_set(map_info_set, 3)
        utm_y = self.tools.get_meta_set(map_info_set, 4)
        ul_x_utm, ul_y_utm = self.tools.get_UTM(pix_e_ul, pix_n_ul,
                                                utm_x, utm_y, self.spy_ps_e,
                                                self.spy_ps_n)
        map_info_set = self.tools.modify_meta_set(map_info_set, 3, ul_x_utm)
        map_info_set = self.tools.modify_meta_set(map_info_set, 4, ul_y_utm)
        metadata['map info'] = map_info_set

        hist_str = (" -> Hyperspectral.crop_single[<"
                    "SpecPyFloatText label: 'pix_e_ul?' value:{0}; "
                    "SpecPyFloatText label: 'pix_n_ul?' value:{1}; "
                    "SpecPyFloatText label: 'pix_e_lr?' value:{2}; "
                    "SpecPyFloatText label: 'pix_n_lr?' value:{3}>]"
                    "".format(pix_e_ul, pix_n_ul, pix_e_lr, pix_n_lr))
        metadata['history'] += hist_str
        metadata['samples'] = array_crop.shape[1]
        metadata['lines'] = array_crop.shape[0]
        self.tools.spyfile.metadata = metadata

        return array_crop, metadata

#        envi.save_image(fname_out_envi + '.hdr', array_img_crop,
#                        dtype=np.float32, force=True, ext=None,
#                        interleave=interleave, metadata=self.metadata)
#
#        fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'
#
#        img_ds = self._get_envi_gdal(fname_in=fname_in)
#        projection_out = img_ds.GetProjection()
#        img_ds = None  # I only want to use GDAL when I have to..
#
#        geotransform_out = [ul_x_utm, self.spy_ps_e, 0.0, ul_y_utm, 0.0,
#                            self.spy_ps_n]
#        self._write_tif(array_img_crop, fname_out_tif, projection_out,
#                        geotransform_out)

#        self.read_cube(fname_in, name_long=name_long, plot_name=plot_name)
#        base_dir_out, name_print = self._save_file_setup(
#                base_dir_out=base_dir_out, folder_name='crop')

#        pix_e_new = pix_e_ul + plot_size_e_pix
#        pix_n_new = pix_n_ul + plot_size_n_pix
#        array_img_crop = self.img_sp.read_subregion((pix_n_ul, pix_n_new),
#                                                    (pix_e_ul, pix_e_new))

#        if name_append is None:
#            name_append = 'crop'
#        name_label = (name_print + self._xstr(name_append) + '.' +
#                      interleave)
#        fname_out_envi = os.path.join(base_dir_out, name_label)
#        print('Spatially cropping image: {0}'.format(name_print))

#        if self.name_plot is not None:
#            fname = (self.name_plot + self._xstr(name_append) + '.' +
#                     interleave)
#        else:
#            fname = (self.name_short + self._xstr(name_append) + '.' +
#                     interleave)
#        fname_out_envi = os.path.join(base_dir_out, fname)

#        print('Cropping plot {0}'.format(self.plot))
#
#        map_info_set = self.metadata['map info']
#        utm_x = self._get_meta_set(map_info_set, 3)
#        utm_y = self._get_meta_set(map_info_set, 4)
#        ul_x_utm, ul_y_utm = self._get_UTM(pix_e_ul, pix_n_ul, utm_x,
#                                           utm_y, size_x=self.spy_ps_e,
#                                           size_y=self.spy_ps_n)
#        map_info_set = self._modify_meta_set(map_info_set, 3, ul_x_utm)
#        map_info_set = self._modify_meta_set(map_info_set, 4, ul_y_utm)
#        self.metadata['map info'] = map_info_set
#        hist_str = (" -> Hyperspectral.crop_single[<"
#                    "SpecPyFloatText label: 'pix_e_ul?' value:{0}; "
#                    "SpecPyFloatText label: 'pix_n_ul?' value:{1} >]"
#                    "".format(pix_e_ul, pix_n_ul))
#        self.metadata['history'] += hist_str
#        self.metadata['samples'] = array_img_crop.shape[1]
#        self.metadata['lines'] = array_img_crop.shape[0]
#        self.metadata['label'] = name_label
#
#        envi.save_image(fname_out_envi + '.hdr', array_img_crop,
#                        dtype=np.float32, force=True, ext=None,
#                        interleave=interleave, metadata=self.metadata)
#
#        fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'
#
##            rgb_list = [self._get_band(640)[0],
##                        self._get_band(550)[0],
##                        self._get_band(460)[0]]
##            from spectral import save_rgb
##            save_rgb(fname_out_tif, array_img_crop, rgb_list, format='tiff')
#
#        img_ds = self._get_envi_gdal(fname_in=fname_in)
#        projection_out = img_ds.GetProjection()
#        img_ds = None  # I only want to use GDAL when I have to..
#
##        drv = gdal.GetDriverByName('ENVI')
##        drv.Register()
##        img_ds = gdal.Open(fname_in, gdalconst.GA_ReadOnly)
##        projection_out = img_ds.GetProjection()
##        img_ds = None
##        drv = None
#
#        geotransform_out = [ul_x_utm, self.spy_ps_e, 0.0, ul_y_utm, 0.0,
#                            self.spy_ps_n]
#        self._write_tif(array_img_crop, fname_out_tif, projection_out,
#                        geotransform_out)

    def load_spyfile(self, spyfile):
        '''
        Loads a `SpyFile` (Spectral Python object) for data access and/or
        manipulation by the `hstools` class.

        Parameters:
            spyfile (`SpyFile` object): The datacube being accessed and/or
                manipulated.
        '''
        self.spyfile = spyfile
        self.tools = hstools(spyfile)
        try:
            self.spy_ps_e = float(self.spyfile.metadata['map info'][5])
            self.spy_ps_n = float(self.spyfile.metadata['map info'][6])
            self.spy_ul_e_srs = float(self.spyfile.metadata['map info'][4])
            self.spy_ul_n_srs = float(self.spyfile.metadata['map info'][3])
        except KeyError as e:
            print('Map information was not able to be loaded from the '
                  '`SpyFile`. Please be sure the metadata contains the "map '
                  'info" tag with accurate geometric information.\n')
            self.spy_ps_e = None
            self.spy_ps_n = None
            self.spy_ul_e_srs = None
            self.spy_ul_n_srs = None
