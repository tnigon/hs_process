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

        self.spy_ps_e = None
        self.spy_ps_n = None
        self.spy_ul_e_srs = None
        self.spy_ul_n_srs = None

        self.defaults = defaults
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
        n_se = srs_n_m - (size_y * n_m)
        n_sw = srs_n_m - (size_y * n_m)
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

    def _find_plots_gdf(self, spyfile, gdf, plot_id_ref, pix_e_ul, pix_n_ul,
                        n_plots, metadata=None):
        '''
        Calculates the number of x plots and y plots in image, determines
        the plot ID number, and calculates and records start/end pixels for
        each plot
        '''
        columns=['plot_id', 'pix_e_ul', 'pix_n_ul', 'crop_e_pix', 'crop_n_pix']
        df_plots = pd.DataFrame(columns=columns)
        gdf_filter = self._overlay_gdf(spyfile, gdf)
        msg = ('Please be sure the reference plot (`plot_id_ref`): {0} is '
               'within the spatial extent of the datacube (`spyfile`):  {1}\n'
               ''.format(plot_id_ref, spyfile.filename))

        # TODO: option to designate any column as the "plot_id" column.
        assert plot_id_ref in gdf_filter['plot'].tolist(), msg

        if metadata is None:
            metadata = spyfile.metadata
#        spy_ps_e = float(metadata['map info'][5])  # pixel size
#        spy_ps_n = float(metadata['map info'][6])
        spy_srs_e_m = float(metadata['map info'][3])  # UTM coordinate
        spy_srs_n_m = float(metadata['map info'][4])

        gdf_temp = (
            gdf_filter.assign(x=lambda df: df['geometry'].centroid.x)
               .assign(y=lambda df: df['geometry'].centroid.y)
        #       .assign(rep_val=lambda df: df[['x', 'y']].mean(axis=1))
#               .sort_values(by=['y', 'x'], ascending=[False, True])
               )
        gdf_temp = gdf_temp.astype({'x': int, 'y': int})
        gdf_sort = gdf_temp.sort_values(by=['y', 'x'], ascending=[False, True])
        gdf_sort = gdf_sort.reset_index(drop=True)  # reset the index

        if pd.notnull(n_plots):
            idx = gdf_sort[gdf_sort['plot'] == plot_id_ref].index[0]
            gdf_sort = gdf_sort.iloc[idx:idx + n_plots]

        for idx, row in gdf_sort.iterrows():
            plot = row['plot']
            bounds = row['geometry'].bounds
            plot_srs_w = bounds[0]
            plot_srs_s = bounds[1]
            plot_srs_e = bounds[2]
            plot_srs_n = bounds[3]
            # plot offset from datacube (from NW/upper left corner)
            offset_e = int((plot_srs_w - spy_srs_e_m) / self.spy_ps_e)
            offset_n = int((spy_srs_n_m - plot_srs_n) / self.spy_ps_n)
            gdf_crop_e_pix = int(abs(plot_srs_e - plot_srs_w) / self.spy_ps_e)
            gdf_crop_n_pix = int(abs(plot_srs_n - plot_srs_s) / self.spy_ps_n)
            data = [plot, offset_e, offset_n, gdf_crop_e_pix, gdf_crop_n_pix]
            df_plots_temp = pd.DataFrame(columns=columns, data=[data])
            df_plots = df_plots.append(df_plots_temp, ignore_index=True)

#        if pix_e_ul is not None:  # compare user-identified pixel to gdf pixel
        if pd.notnull(pix_e_ul):  # compare user-identified pixel to gdf pixel
            gdf_e = df_plots[df_plots[
                    'plot_id'] == plot_id_ref]['pix_e_ul'].item()
            delta_e = pix_e_ul - gdf_e
        else:
            delta_e = 0
            # positive means error of image georeferenced to the right/E
        if pd.notnull(pix_n_ul):  # compare user-identified pixel to gdf pixel
            gdf_n = df_plots[df_plots[
                    'plot_id'] == plot_id_ref]['pix_n_ul'].item()
            # positive means error of image georeferenced to the bottom/S
            delta_n = pix_n_ul - gdf_n
        else:
            delta_n = 0

        for idx, row in df_plots.iterrows():
            plot_id = row['plot_id']
            gdf_e = row['pix_e_ul']
            shft_e = gdf_e + delta_e  # if `delta_e` is positive, move right/E
            df_plots.loc[df_plots['plot_id'] == plot_id, 'pix_e_ul'] = shft_e
            gdf_n = row['pix_n_ul']
            shft_n = gdf_n + delta_n  # if `delta_n` is positive, move up/N
            df_plots.loc[df_plots['plot_id'] == plot_id, 'pix_n_ul'] = shft_n
        return df_plots

    def _record_pixels(self, plot_id_ul, plot_n_start, plot_n_end, row_plot,
                       df_plots, crop_e_pix, crop_n_pix, pix_e_ul,
                       pix_n_ul, experiment='wells'):
        '''

        '''
        if experiment == 'wells':
            for plot_n in range(plot_n_start, plot_n_end):  # top plots
                col_plot = (plot_n) % 5
                if col_plot == 0:
                    row_plot += 1
                plot_id = plot_id_ul - (col_plot * 100)  # E/W
                plot_id = plot_id - row_plot  # N/S
                col_pix = (col_plot * crop_e_pix) + pix_e_ul
                row_pix = (row_plot * crop_n_pix) + pix_n_ul

#                array_crop, metadata = self.crop_single(
#                    col_pix, row_pix, crop_e_pix,
#                    crop_n_pix)  # lines and samples backwards in metadata

                df_temp = pd.DataFrame(data=[[self.spyfile.filename, plot_id,
                                              col_plot, row_plot,
                                              col_pix, row_pix]],
                                       columns=df_plots.columns)
                df_plots = df_plots.append(df_temp, ignore_index=True)

            return df_plots, row_plot

    def _calc_size(self, plot_id_ul, n_plots_x, row_plots_top, row_plots_bot,
                   crop_e_pix, crop_n_pix, pix_e_ul, pix_n_ul):
        '''
        Calculates the number of x plots and y plots in image, determines
        the plot ID number, and calculates and records start/end pixels for
        each plot
        '''
#        df_plots = pd.DataFrame(columns=['plot_id', 'col_plot',
#                                         'row_plot', 'col_pix', 'row_pix',
#                                         'array_crop', 'metadata'])
        df_plots = pd.DataFrame(columns=['fname_in', 'plot_id', 'col_plot',
                                         'row_plot', 'pix_e_ul', 'pix_n_ul'])
        row_plot = -1
        plot_n_start = 0
        plot_n_end = n_plots_x * row_plots_top

        df_plots, row_plot = self._record_pixels(
                plot_id_ul, plot_n_start, plot_n_end, row_plot, df_plots,
                crop_e_pix, crop_n_pix, pix_e_ul, pix_n_ul)

        if row_plots_bot > 0:  # do the same for bottom, adjusting start/end
            plot_n_start = plot_n_end
            plot_n_bot = n_plots_x * row_plots_bot
            plot_n_end = plot_n_end + plot_n_bot
            df_plots, row_plot = self._record_pixels(
                    plot_id_ul, plot_n_start, plot_n_end, row_plot, df_plots,
                    crop_e_pix, crop_n_pix, pix_e_ul, pix_n_ul)

        return df_plots

    def _check_alley(self, plot_id_ul, n_plots_y, rows_pix,
                     pix_n_ul, crop_n_pix, alley_size_pix):
        '''
        Calculates whether there is an alleyway in the image (based on plot
        configuration), then adjusts n_plots_y so it is correct after
        considering the alley

        rows_pix (`int`): number of pixel rows in image
        '''
        plot_id_tens = abs(plot_id_ul) % 100
        row_plots_top = plot_id_tens % n_plots_y
        if row_plots_top == 0:  # gets number of plots left in block
            row_plots_top = n_plots_y  # remainder is 0, not 9..

        # we have plots left until alley, but image may not extend that far
        pix_avail = rows_pix - abs(pix_n_ul)  # number of pixels south of ul
        row_plots_avail = int(pix_avail / crop_n_pix)  # gets number of whole plots
        if row_plots_top > row_plots_avail:
            row_plots_top = row_plots_avail

        if row_plots_top < row_plots_avail:  # gets remaining pixels south of block
            pix_remain = (pix_avail - (row_plots_top * abs(crop_n_pix)))
        else:  # no plots at the bottom block, just at the top block
            row_plots_bot = 0
            return row_plots_top, row_plots_bot

        if pix_remain >= abs(alley_size_pix + crop_n_pix):
            # have room for more plots (must still remove 2 rows of plots)
            # calculate rows remain after skip
            row_plots_bot = int(abs((pix_remain + alley_size_pix) /
                                     crop_n_pix))
#            n_plots_y = row_plots_top + row_plots_bot
        # these are for if alley_size_pix is large but crop_n_pix is relatively small..
        else:
#            n_plots_y = row_plots_top
            row_plots_bot = 0
#        elif pix_remain >= abs(crop_n_pix) * 2:
#            # remove 2 rows of plots
#            n_plots_y -= 2
#        elif pix_remain >= abs(crop_n_pix):
#            # remove 1 row of plots
#            n_plots_y -= 1
#        else:
#            # works out perfect.. don't have to change anything
#            pass
        return row_plots_top, row_plots_bot


    def _get_corners(self, pix_ul, crop_pix, buf_pix):
        '''
        Gets the upper left and lower right corner of the cropped array. If
        necessary, applies the buffer to the coordinates. This is a generic
        function that can be used in either the easting or northing direction.

        Parameters:
            pix_ul (`int`): upper left pixel coordinate as an index (first
                pixel is zero; can be either easting or northing direction).
            crop_pix (`int`): number of pixels to be cropped (before applying
                the buffer).
            buf_pix (`int`): number of pixels to buffer

        Returns:
            2-element `tuple` containing

            - **pix_ul** (`int`): upper left pixel coordinate after applying the
              buffer.
            - **pix_lr** (`int`): lower right pixel coordinate after applying the
              buffer.
        '''
        pix_lr = pix_ul + crop_pix
        if buf_pix is not None:
            pix_ul += buf_pix
            pix_lr -= buf_pix
        return pix_ul, pix_lr

    def _handle_defaults(self, e_pix, n_pix, e_m, n_m, group='crop',
                         spyfile=None):
        '''
        If these are set to `None`, retrieves default values from
        `spatial_mod.defaults`, which can be accessed and modified by an
        instance of this class by a higher level program. Also converts
        betweeen pixel units and map units if one is populated and the other is
        `None`.
        '''
        if not isinstance(spyfile, SpyFile.SpyFile):
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        if group == 'crop':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults['crop_e_pix']
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults['crop_n_pix']
        elif group == 'alley':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults['alley_size_e_pix']
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults['alley_size_n_pix']
        elif group == 'buffer':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults['buf_e_pix']
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults['buf_n_pix']

        if pd.isnull(e_pix) and pd.notnull(e_m):
            e_pix = int(round(e_m / self.spy_ps_e))
        elif pd.notnull(e_pix) and pd.isnull(e_m):
            e_m = e_pix * self.spy_ps_e
        if pd.isnull(n_pix) and pd.notnull(n_m):
            n_pix = int(round(n_m / self.spy_ps_n))
        elif pd.notnull(n_pix) and pd.isnull(n_m):
            n_m = n_pix * self.spy_ps_n
        return e_pix, n_pix, e_m, n_m

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

    def _pix_to_mapunit(self, e_m, n_m, e_pix, n_pix, ps_e=None, ps_n=None):
        '''
        Converts between pixel units and map units (e.g., UTM meters).

        Parameters:
            e_m (`float`): easting map unit coordinate.
            n_m (`float`): northing map unit coordinate.
            e_pix (`int`): easting pixel coordinate.
            n_pix (`int`): northing pixel coordinate.
        '''
        if ps_e is None:
            ps_e = self.spy_ps_e
        if ps_n is None:
            ps_n = self.spy_ps_n

        if e_pix is None and e_m is not None:
            e_pix = int(e_m / ps_e)
        elif e_pix is not None and e_m is None:
            e_m = e_pix * ps_e
        if n_pix is None and n_m is not None:
            n_pix = int(n_m / ps_n)
        elif n_pix is not None and n_m is None:
            n_m = n_pix * ps_n
        return e_m, n_m, e_pix, n_pix

    def _shift_by_gdf(self, gdf, plot_id, buf_e_m, buf_n_m):
        '''
        Applies a shift to the geotransform of a plot based on its location as
        determined by the geometry of the `geopandas.GeoDataFrame`. This
        effectively centers each cropped datacube within its plot boundary.

        Parameters:
            df_plots (pandas.DataFrame):
            gdf (geopandas.GeoDataFrame):
            plot_id
            buf_e_m
            buf_n_m
        '''
        gdf_plot = gdf[gdf['plot'] == plot_id]
        if pd.isnull(buf_e_m):
            buf_e_m = 0
        if pd.isnull(buf_n_m):
            buf_n_m = 0
        ul_x_utm = gdf_plot['geometry'].bounds['minx'].item() + buf_e_m
        ul_y_utm = gdf_plot['geometry'].bounds['maxy'].item() - buf_n_m
        return ul_x_utm, ul_y_utm

    def crop_many_gdf(self, plot_id_ref, pix_e_ul=0, pix_n_ul=0,
                      crop_e_m=None, crop_n_m=None, crop_e_pix=None,
                      crop_n_pix=None, buf_e_m=None, buf_n_m=None,
                      buf_e_pix=None, buf_n_pix=None, n_plots=None,
                      spyfile=None, gdf=None):
        '''
        Crops many plots from a single image by comparing the image to a
        polygon file (geopandas.GoeDataFrame) that contains plot information
        and geometry of plot boundaries.

        If `pix_e_ul` and `pix_n_ul` are passed, those pixel offests will be
        compared to the upper left pixel of `plot_id_ref` and the difference in
        both the easting and northing direction will be applied to all plots to
        systematically shift the actual upper left pixel/cropping location for
        each plot within the image.

        Parameters:
            plot_id_ref (`int`): the plot ID of the reference plot, which must
                be present in the datacube.
            pix_e_ul (`int`): upper left pixel column (easting) of
                `plot_id_ref`; this is used to calculate the offset between the
                GeoDataFrame geometry and the approximate image georeference
                error (default: 0).
            pix_n_ul (`int`): upper left pixel row (northing) of `plot_id_ref`;
                this is used to calculate the offset between the GeoDataFrame
                geometry and the approximate image georeference error
                (default: 0).
            crop_e_m (`float`, optional): length of each row (easting
                direction) in the cropped image (in map units; e.g., meters).
            crop_n_m (`float`, optional): length of each column (northing
                direction) in the cropped image (in map units; e.g., meters).
            crop_e_pix (`int`, optional): number of pixels in each row in the
                cropped image.
            crop_n_pix (`int`, optional): number of pixels in each column in
                the cropped image.
            buf_e_m (`float`, optional): The buffer distance in the easting
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after `crop_X_m`/`crop_X_pix`. A positive value will reduce the
                size of `crop_X_m`/`crop_X_pix`, and a negative value will
                increase it.
            buf_n_m (`float`, optional): The buffer distance in the northing
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after `crop_X_m`/`crop_X_pix`. A positive value will reduce the
                size of `crop_X_m`/`crop_X_pix`, and a negative value will
                increase it.
            buf_e_pix (`int`, optional): The buffer distance in the easting
                direction (in pixel units) to be applied after calculating the
                original crop area.
            buf_n_pix (`int`, optional): The buffer distance in the northing
                direction (in pixel units) to be applied after calculating the
                original crop area.
            n_plots (`int`, optional): number of plots to crop, starting with
                `plot_id_ref` and moving from West to East and North to South
                (default; `None`).
           spyfile (`SpyFile` object): The datacube to crop; if `None`, loads
               datacube and band information from `spatial_mod.spyfile`
               (default: `None`).
            gdf (`geopandas.GeoDataFrame`): the plot names and polygon
                geometery of each of the plots; 'plot' must be used as a column
                name to identify each of the plots, and should be an integer.

        Returns:
            `pandas.DataFrame`:
                - **df_plots** (`pandas.DataFrame`) -- data for
                  which to crop each plot; includes 'plot_id', 'pix_e_ul', and
                  'pix_n_ul' columns. This data can be passed to
                  `spatial_mod.crop_single()` to perform the actual cropping.

        Note:
            Either the pixel coordinate or the map unit coordinate should be
            passed for `crop_X_Y` and `buf_X_Y` in each direction (i.e.,
            easting and northing). Do not pass both.
        '''
        if spyfile is None:
            spyfile = self.spyfile
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
        metadata = self.spyfile.metadata
        if gdf is None:
            gdf = self.gdf

        msg1 = ('Please load a GeoDataFrame (geopandas library).\n')
        msg2 = ('Please be sure `plot_id_ref` is present in `gdf` (i.e., '
                'the GeoDataFrame).\n')
        assert isinstance(gdf, gpd.GeoDataFrame), msg1
        assert plot_id_ref in gdf['plot'].tolist(), msg2
        df_plots = self._find_plots_gdf(spyfile, gdf, plot_id_ref,
                                        pix_e_ul, pix_n_ul, n_plots, metadata)
        # TODO: add crop_X and buf_X to df_plots
        return df_plots

    def crop_many_grid(self, plot_id_ul, pix_e_ul, pix_n_ul,
                       crop_e_m=9.170, crop_n_m=3.049,
                       crop_e_pix=None, crop_n_pix=None,
                       buf_e_pix=None, buf_n_pix=None,
                       buf_e_m=None, buf_n_m=None,
                       alley_size_e_m=None, alley_size_n_m=None,
                       alley_size_e_pix=None, alley_size_n_pix=None,
                       n_plots_x=5, n_plots_y=9, spyfile=None):
        '''
        Crops many plots from a single image by calculating the distance
        between plots based on `crop_X_Y`, `n_plots_X/Y`, and `alley_size_X_Y`
        parameters.

        Parameters:
            plot_id_ul (`int`): the plot ID of the upper left (northwest-most)
                plot to be cropped.
            pix_e_ul (`int`): upper left pixel column (easting) of
                `plot_id_ul`.
            pix_n_ul (`int`): upper left pixel row (northing) of `plot_id_ul`.
            crop_e_m (`float`, optional): length of each row (easting
                direction) in the cropped image (in map units; e.g., meters).
            crop_n_m (`float`, optional): length of each column (northing
                direction) in the cropped image (in map units; e.g., meters).
            crop_e_pix (`int`, optional): number of pixels in each row in the
                cropped image.
            crop_n_pix (`int`, optional): number of pixels in each column in
                the cropped image.
            buf_e_m (`float`, optional): The buffer distance in the easting
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after `crop_X_m`/`crop_X_pix`. A positive value will reduce the
                size of `crop_X_m`/`crop_X_pix`, and a negative value will
                increase it.
            buf_n_m (`float`, optional): The buffer distance in the northing
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after `crop_X_m`/`crop_X_pix`. A positive value will reduce the
                size of `crop_X_m`/`crop_X_pix`, and a negative value will
                increase it.
            buf_e_pix (`int`, optional): The buffer distance in the easting
                direction (in pixel units) to be applied after calculating the
                original crop area.
            buf_n_pix (`int`, optional): The buffer distance in the northing
                direction (in pixel units) to be applied after calculating the
                original crop area.
            alley_size_e_m (`int`, optional): Should be passed if there are
                alleys passing across the E/W direction of the plots that are
                not accounted for by the `crop_X_Y` parameters. Used together
                with `n_plots_x` to determine the plots represented in a
                particular datacube.
            alley_size_n_m (`int`, optional): Should be passed if there are
                alleys passing across the N/S direction of the plots that are
                not accounted for by the `crop_X_Y` parameters. Used together
                with `n_plots_y` to determine the plots represented in a
                particular datacube.
            alley_size_e_pix (`float`, optional): see `alley_size_e_m`.
            alley_size_n_pix (`float`, optional): see `alley_size_n_m`.
            n_plots_x (`int`): number of plots in a row in east/west direction
                (default: 5).
            n_plots_y (`int`): number of plots in a row in north/south
                direction (default: 9).
            spyfile (`SpyFile` object): The datacube to crop; if `None`, loads
                datacube and band information from `spatial_mod.spyfile`
                (default: `None`).

        Returns:
            `pandas.DataFrame`:
                - **df_plots** (`pandas.DataFrame`) -- data for
                  which to crop each plot; includes 'plot_id', 'pix_e_ul', and
                  'pix_n_ul' columns. This data can be passed to
                  `spatial_mod.crop_single()` to perform the actual cropping.

        Note:
            Either the pixel coordinate or the map unit coordinate should be
            passed for `crop_X_Y` and `buf_X_Y` in each direction (i.e.,
            easting and northing). Do not pass both.
        '''
        if spyfile is None:
            spyfile = self.spyfile
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)

        msg1 = ('Either crop_size_XX_m or crop_size_XX_pix should be passed. '
                'Please pass one or the other.')
        msg2 = ('Either crop_size_XX_m or crop_size_XX_pix should be passed. '
                'Do not pass both.')
        assert not all(
                v is None for v in [crop_e_m, crop_e_pix]), msg1
        assert not all(
                v is not None for v in [crop_e_m, crop_e_pix]), msg2

        crop_e_m, crop_n_m, crop_e_pix, crop_n_pix = self._pix_to_mapunit(
                crop_e_m, crop_n_m, crop_e_pix, crop_n_pix)
        buf_e_m, buf_n_m, buf_e_pix, buf_n_pix = self._pix_to_mapunit(
                buf_e_m, buf_n_m, buf_e_pix, buf_n_pix)
        alley_size_e_m, alley_size_n_m, alley_size_e_pix, alley_size_e_pix =\
            self._pix_to_mapunit(alley_size_e_m, alley_size_n_m,
                                 alley_size_e_pix, alley_size_e_pix)

        row_plots_top, row_plots_bot = self._check_alley(
                plot_id_ul, n_plots_y, spyfile.nrows, pix_n_ul,
                crop_n_pix, alley_size_n_pix)
        df_plots = self._calc_size(plot_id_ul, n_plots_x, row_plots_top,
                                   row_plots_bot, crop_e_pix,
                                   crop_n_pix, pix_e_ul, pix_n_ul)
        return df_plots

    def crop_single(self, pix_e_ul, pix_n_ul, crop_e_pix=None,
                    crop_n_pix=None, crop_e_m=None, crop_n_m=None,
                    buf_e_pix=None, buf_n_pix=None, buf_e_m=None, buf_n_m=None,
                    spyfile=None, plot_id=None, gdf=None,
                    name_append='spatial-crop-single'):
        '''
        Crops a single plot from an image. If `plot_id` and `gdf` are
        explicitly passed (i.e., they will not be loaded from `spatial_mod`
        class), the "map info" tag in the metadata will be adjusted to center
        the cropped area within the appropriate plot geometry.

        Parameters:
            pix_e_ul (`int`): upper left column (easting) to begin cropping
            pix_n_ul (`int`): upper left row (northing) to begin cropping
            crop_e_pix (`int`, optional): number of pixels in each row in the
                cropped image
            crop_n_pix (`int`, optional): number of pixels in each column in
                the cropped image
            crop_e_m (`float`, optional): length of each row (easting
                     direction) in the cropped image (in map units; e.g.,
                     meters).
            crop_n_m (`float`, optional): length of each column (northing
                     direction) in the cropped image (in map units; e.g.,
                     meters).
            buf_e_pix (`int`, optional): applied/calculated after
                `crop_e_pix` and `crop_n_pix` are accounted for
            buf_n_pix (`int`, optional):
            buf_e_m (`float`, optional):
            buf_n_m (`float`, optional):
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
            plot_id (`int`): the plot ID of the area to be cropped.
            gdf (`geopandas.GeoDataFrame`): the plot names and polygon
                geometery of each of the plots; 'plot' must be used as a column
                name to identify each of the plots, and should be an integer.
                `gdf` must be explicitly passed to
            name_append (`str`): NOT YET SUPPORTED; name to append to the
                filename (default: 'spatial-crop-single').

        Returns:
            2-element `tuple` containing

            - **array_crop** (`numpy.ndarray`): Cropped datacube.
            - **metadata** (`dict`): Modified metadata describing the cropped
              hyperspectral datacube (`array_crop`).
        '''
        crop_e_pix, crop_n_pix, crop_e_m, crop_n_m = self._handle_defaults(
                crop_e_pix, crop_n_pix, crop_e_m, crop_n_m, group='crop')
        buf_e_pix, buf_n_pix, buf_e_m, buf_n_m = self._handle_defaults(
                buf_e_pix, buf_n_pix, buf_e_m, buf_n_m, group='buffer')
        pix_e_ul, pix_e_lr = self._get_corners(pix_e_ul, crop_e_pix,
                                               buf_e_pix)
        pix_n_ul, pix_n_lr = self._get_corners(pix_n_ul, crop_n_pix,
                                               buf_n_pix)
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

        if isinstance(gdf, gpd.GeoDataFrame) and plot_id is not None:
            ul_x_utm, ul_y_utm = self._shift_by_gdf(gdf, plot_id,
                                                    buf_e_m, buf_n_m)
        else:
            utm_x = self.tools.get_meta_set(map_info_set, 3)
            utm_y = self.tools.get_meta_set(map_info_set, 4)
            ul_x_utm, ul_y_utm = self.tools.get_UTM(pix_e_ul, pix_n_ul,
                                                    utm_x, utm_y,
                                                    self.spy_ps_e,
                                                    self.spy_ps_n)

        map_info_set = self.tools.modify_meta_set(map_info_set, 3, ul_x_utm)
        map_info_set = self.tools.modify_meta_set(map_info_set, 4, ul_y_utm)
        metadata['map info'] = map_info_set

        hist_str = (" -> hs_process.crop_single[<"
                    "SpecPyFloatText label: 'pix_e_ul?' value:{0}; "
                    "SpecPyFloatText label: 'pix_n_ul?' value:{1}; "
                    "SpecPyFloatText label: 'pix_e_lr?' value:{2}; "
                    "SpecPyFloatText label: 'pix_n_lr?' value:{3}>]"
                    "".format(pix_e_ul, pix_n_ul, pix_e_lr, pix_n_lr))
        # If "..crop_single" is already included in the history, remove it
        idx_remove = metadata['history'].find(
                ' -> hs_process.crop_single[<')
        if idx_remove == -1:
            metadata['history'] += hist_str
        else:
            metadata['history'] = metadata['history'][:idx_remove]
            metadata['history'] += hist_str
        metadata['samples'] = array_crop.shape[1]
        metadata['lines'] = array_crop.shape[0]

        # TODO: Figure out if/when the 'label' tag should be changed.
#        label = metadata['label']
#        if label is not None:
#            name_label = (os.path.splitext(label)[0] + '-' + name_append + '.'
#                          + self.defaults.interleave)
#        metadata['label'] = name_label
        self.tools.spyfile.metadata = metadata

        return array_crop, metadata

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
            self.spy_ul_e_srs = float(self.spyfile.metadata['map info'][3])
            self.spy_ul_n_srs = float(self.spyfile.metadata['map info'][4])
            self.spy_ps_e = float(self.spyfile.metadata['map info'][5])
            self.spy_ps_n = float(self.spyfile.metadata['map info'][6])
        except KeyError as e:
            print('Map information was not able to be loaded from the '
                  '`SpyFile`. Please be sure the metadata contains the "map '
                  'info" tag with accurate geometric information.\n')
            self.spy_ul_e_srs = None
            self.spy_ul_n_srs = None
            self.spy_ps_e = None
            self.spy_ps_n = None
