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
    Class for manipulating data within the spatial domain
    (e.g., cropping a datacube by a geographical boundary).
    '''
    def __init__(self, spyfile, gdf=None):
        '''
        spyfile (``SpyFile`` object): The Spectral Python datacube to manipulate.
        gdf (``geopandas.DataFrame``): Polygon data that includes the plot_id and
            its geometry.
        '''
        self.spyfile = spyfile
        self.gdf = gdf

        self.spy_ps_e = None
        self.spy_ps_n = None
        self.spy_ul_e_srs = None
        self.spy_ul_n_srs = None
        self.tools = None


        self.defaults = defaults()
        self.load_spyfile(spyfile)

    def _create_spyfile_extent_gdf(self, epsg=32615):
        '''
        '''
        metadata = self.spyfile.metadata
        crs = 'epsg:{0}'.format(epsg)
        size_x = self.spyfile.shape[1]  # number of pixels
        size_y = self.spyfile.shape[0]

        e_nw = self.spy_ul_e_srs
        e_ne = self.spy_ul_e_srs + (size_x * self.spy_ps_e)
        e_se = self.spy_ul_e_srs + (size_x * self.spy_ps_e)
        e_sw = self.spy_ul_e_srs
        n_nw = self.spy_ul_n_srs
        n_ne = self.spy_ul_n_srs
        n_se = self.spy_ul_n_srs - (size_y * self.spy_ps_n)
        n_sw = self.spy_ul_n_srs - (size_y * self.spy_ps_n)
        coords_e = [e_nw, e_ne, e_se, e_sw, e_nw]
        coords_n = [n_nw, n_ne, n_se, n_sw, n_nw]

        polygon_geom = Polygon(zip(coords_e, coords_n))
        gdf_sp = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
        return gdf_sp

    def _overlay_gdf(self, gdf, epsg_sp=32615, how='intersection', crop=True):
        '''
        Performs a geopandas overlay between the input geodatafram (``gdf``) and
        the extent of ``spyfile``.

        If crop is ``True``, then the polygon bounds will be cropped to the
        extent of the overlay based on the ``how`` parameter.
        '''
        gdf_sp = self._create_spyfile_extent_gdf(epsg=epsg_sp)
        gdf_filter = gpd.overlay(gdf, gdf_sp, how=how)
        if crop is not True:  # simply return the original plot bounds that intersect
            gdf_filter = gdf[gdf['plot_id'].isin(gdf_filter['plot_id'])].reset_index(drop=True)
        return gdf_filter

    def _find_plots_gdf(self, gdf, plot_id_ref, pix_e_ul, pix_n_ul,
                        n_plots, metadata=None):
        '''
        Calculates the number of x plots and y plots in image, determines
        the plot ID number, and calculates and records start/end pixels for
        each plot
        '''
        columns = [self.defaults.spat_crop_cols.directory,
                   self.defaults.spat_crop_cols.name_short,
                   self.defaults.spat_crop_cols.name_long,
                   self.defaults.spat_crop_cols.ext,
                   self.defaults.spat_crop_cols.plot_id_ref,
                   self.defaults.spat_crop_cols.pix_e_ul,
                   self.defaults.spat_crop_cols.pix_n_ul,
                   self.defaults.spat_crop_cols.buf_e_m,
                   self.defaults.spat_crop_cols.buf_n_m,
                   self.defaults.spat_crop_cols.buf_e_pix,
                   self.defaults.spat_crop_cols.buf_n_pix,
                   self.defaults.spat_crop_cols.crop_e_m,
                   self.defaults.spat_crop_cols.crop_n_m,
                   self.defaults.spat_crop_cols.crop_e_pix,
                   self.defaults.spat_crop_cols.crop_n_pix,
                   self.defaults.spat_crop_cols.gdf_shft_e_m,
                   self.defaults.spat_crop_cols.gdf_shft_n_m,
                   self.defaults.spat_crop_cols.gdf_shft_e_pix,
                   self.defaults.spat_crop_cols.gdf_shft_n_pix]
        # gdf_shft_e_m and gdf_shft_n_m are considered when df_plots is passed to crop_single()

        df_plots = pd.DataFrame(columns=columns)
        gdf_filter = self._overlay_gdf(gdf, crop=False)
        msg = ('Please be sure the reference plot (`plot_id_ref`) is within '
               'the spatial extent of the datacube (`spyfile`).\nCurrent '
               'value of `plot_id_ref`: {0}\nDatacube filename: {1}\n'
               ''.format(plot_id_ref, self.spyfile.filename))
        if pix_e_ul == 0:
            pix_e_ul = None
        if pix_n_ul == 0:
            pix_n_ul = None
        # if pd.notnull(n_plots) or pd.notnull(pix_e_ul) or pd.notnull(pix_n_ul):
        if pd.notnull(pix_e_ul) or pd.notnull(pix_n_ul):
            if pd.notnull(plot_id_ref):
                assert plot_id_ref in gdf_filter['plot_id'].tolist(), msg
            else:
                raise ValueError(
                    '``plot_id_ref`` was not passsed, and is required if '
                    'are ``pix_e_ul`` or ``pix_n_ul`` are passed.')
        # TODO: option to designate any column as the "plot_id" column.

        if metadata is None:
            metadata = self.spyfile.metadata
#        spy_ps_e = float(metadata['map info'][5])  # pixel size
#        spy_ps_n = float(metadata['map info'][6])
        spy_srs_e_m = float(metadata['map info'][3])  # UTM coordinate
        spy_srs_n_m = float(metadata['map info'][4])

        # spy_srs_e_m = float(self.tools.get_meta_set(metadata['map info'])[3])
        # spy_srs_n_m = float(self.tools.get_meta_set(metadata['map info'])[4])

        gdf_temp = (
            gdf_filter.assign(x=lambda df: df['geometry'].centroid.x)
               .assign(y=lambda df: df['geometry'].centroid.y)
        #       .assign(rep_val=lambda df: df[['x', 'y']].mean(axis=1))
#               .sort_values(by=['y', 'x'], ascending=[False, True])
               )
        gdf_temp = gdf_temp.astype({'x': int, 'y': int})
        gdf_sort = gdf_temp.sort_values(by=['y', 'x'], ascending=[False, True])
        gdf_sort = gdf_sort.reset_index(drop=True)  # reset the index

        if pd.notnull(n_plots) and pd.isnull(plot_id_ref):
            plot_id_ref = gdf_sort.loc[0,'plot_id']  # Get NW-most plot_id as plot_id_ref
        if pd.notnull(n_plots):
            idx = gdf_sort[gdf_sort['plot_id'] == plot_id_ref].index[0]
            gdf_sort = gdf_sort.iloc[idx:idx + int(n_plots)]

        for idx, row in gdf_sort.iterrows():
            plot_id = row['plot_id']
            bounds = row['geometry'].bounds
            plot_srs_w = bounds[0]
            plot_srs_s = bounds[1]
            plot_srs_e = bounds[2]
            plot_srs_n = bounds[3]
            # plot offset from datacube (from NW/upper left corner)
            offset_e = int((plot_srs_w - self.spy_ul_e_srs) / self.spy_ps_e)
            offset_n = int((self.spy_ul_n_srs - plot_srs_n) / self.spy_ps_n)

            gdf_crop_e_pix = int(abs(plot_srs_e - plot_srs_w) / self.spy_ps_e)
            gdf_crop_n_pix = int(abs(plot_srs_n - plot_srs_s) / self.spy_ps_n)
            data = [self.tools.base_dir,
                    self.tools.name_short,
                    self.tools.name_long,
                    os.path.splitext(self.spyfile.filename)[-1],
                    plot_id, offset_e, offset_n,
                    np.nan, np.nan, np.nan, np.nan,  # buf_X
                    np.nan, np.nan, gdf_crop_e_pix, gdf_crop_n_pix,  # crop
                    np.nan, np.nan, np.nan, np.nan]  # gdf_shft]
            df_plots_temp = pd.DataFrame(columns=columns, data=[data])
            # TODO: Check array size and delete if there is no non-nan pixels
            df_plots = df_plots.append(df_plots_temp, ignore_index=True)

#        if pix_e_ul is not None:  # compare user-identified pixel to gdf pixel
        if pd.notnull(pix_e_ul):  # compare user-identified pixel to gdf pixel
            gdf_e = df_plots[df_plots[
                    'plot_id_ref'] == plot_id_ref]['pix_e_ul'].item()
            delta_e = pix_e_ul - gdf_e
        else:
            delta_e = 0
            # positive means error of image georeferenced to the right/E
        if pd.notnull(pix_n_ul):  # compare user-identified pixel to gdf pixel
            gdf_n = df_plots[df_plots[
                    'plot_id_ref'] == plot_id_ref]['pix_n_ul'].item()
            # positive means error of image georeferenced to the bottom/S
            delta_n = pix_n_ul - gdf_n
        else:
            delta_n = 0

        # print('delta_e: {0}'.format(delta_e))
        # print('delta_n: {0}'.format(delta_n))
        for idx, row in df_plots.iterrows():
            plot_id_ref = row['plot_id_ref']
            gdf_e = row['pix_e_ul']
            shft_e = gdf_e + delta_e  # if `delta_e` is positive, move right/E
            df_plots.loc[df_plots['plot_id_ref'] == plot_id_ref, 'pix_e_ul'] = shft_e
            gdf_n = row['pix_n_ul']
            shft_n = gdf_n + delta_n  # if `delta_n` is positive, move up/N
            df_plots.loc[df_plots['plot_id_ref'] == plot_id_ref, 'pix_n_ul'] = shft_n
            # print('Plot: {0}'.format(row['plot_id_ref']))
            # print('gdf_e: {0}'.format(gdf_e))
            # print('delta_e: {0}'.format(delta_e))
            # print('delta_n: {0}'.format(row['plot_id_ref']))
        # if we don't actually crop and write the datacube here, we have to pass
        # shft_e and shft_n so the metadata can be adjusted during/after the
        # actual cropping.

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
        df_plots = pd.DataFrame(columns=['fname_in', 'plot_id_ref', 'col_plot',
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

        rows_pix (``int``): number of pixel rows in image
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
            pix_ul (``int``): upper left pixel coordinate as an index (first
                pixel is zero; can be either easting or northing direction).
            crop_pix (``int``): number of pixels to be cropped (before applying
                the buffer).
            buf_pix (``int``): number of pixels to buffer

        Returns:
            2-element ``tuple`` containing

            - **pix_ul** (``int``): upper left pixel coordinate after applying the
              buffer.
            - **pix_lr** (``int``): lower right pixel coordinate after applying the
              buffer.
        '''
        pix_lr = pix_ul + crop_pix
        if buf_pix is not None:
            pix_ul += buf_pix
            pix_lr -= buf_pix
        return int(pix_ul), int(pix_lr)

    def _handle_defaults(self, e_pix, n_pix, e_m, n_m, group='crop',
                         spyfile=None):
        '''
        If these are set to ``None``, retrieves default values from
        ``spatial_mod.defaults``, which can be accessed and modified by an
        instance of this class by a higher level program. Also converts
        betweeen pixel units and map units if one is populated and the other is
        ``None``.
        '''
        if not isinstance(spyfile, SpyFile.SpyFile):
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        if group == 'crop':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults.crop_e_pix
                e_m = self.defaults.crop_defaults.crop_e_m
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults.crop_n_pix
                n_m = self.defaults.crop_defaults.crop_n_m
        elif group == 'alley':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults.alley_size_e_pix
                e_m = self.defaults.crop_defaults.alley_size_e_m
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults.alley_size_n_pix
                n_m = self.defaults.crop_defaults.alley_size_n_m
        elif group == 'buffer':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults.buf_e_pix
                e_m = self.defaults.crop_defaults.buf_e_m
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults.buf_n_pix
                n_m = self.defaults.crop_defaults.buf_n_m
        elif group == 'gdf_shft':
            if pd.isnull(e_pix) and pd.isnull(e_m):
                e_pix = self.defaults.crop_defaults.gdf_shft_e_pix
                e_m = self.defaults.crop_defaults.gdf_shft_e_m
            if pd.isnull(n_pix) and pd.isnull(n_m):
                n_pix = self.defaults.crop_defaults.gdf_shft_n_pix
                n_m = self.defaults.crop_defaults.gdf_shft_n_m

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
            plot_id = feat.GetField('plot_id')
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
            e_m (``float``): easting map unit coordinate.
            n_m (``float``): northing map unit coordinate.
            e_pix (``int``): easting pixel coordinate.
            n_pix (``int``): northing pixel coordinate.
        '''
        if ps_e is None:
            ps_e = self.spy_ps_e
        if ps_n is None:
            ps_n = self.spy_ps_n

        if pd.isnull(e_pix) and pd.notnull(e_m):
            e_pix = int(e_m / ps_e)
        elif pd.notnull(e_pix) and pd.isnull(e_m):
            e_m = e_pix * ps_e
        if pd.isnull(n_pix) and pd.notnull(n_m):
            n_pix = int(n_m / ps_n)
        elif pd.notnull(n_pix) and pd.isnull(n_m):
            n_m = n_pix * ps_n
        return e_m, n_m, e_pix, n_pix

    def _shift_by_gdf(self, gdf, plot_id_ref, buf_e_m, buf_n_m,
                      gdf_shft_e_m, gdf_shft_n_m):
        '''
        Applies a shift to the geotransform of a plot based on its location as
        determined by the geometry of the ``geopandas.GeoDataFrame``. This
        effectively centers each cropped datacube within its plot boundary.

        Parameters:
            df_plots (pandas.DataFrame):
            gdf (geopandas.GeoDataFrame):
            plot_id
            buf_e_m
            buf_n_m
        '''
        gdf_plot = gdf[gdf['plot_id'] == plot_id_ref]
        if pd.isnull(buf_e_m):
            buf_e_m = 0
        if pd.isnull(buf_n_m):
            buf_n_m = 0
        if pd.isnull(gdf_shft_e_m):
            gdf_shft_e_m = 0
        if pd.isnull(gdf_shft_n_m):
            gdf_shft_n_m = 0
        ul_x_utm = gdf_plot['geometry'].bounds['minx'].item() + buf_e_m + gdf_shft_e_m
        ul_y_utm = gdf_plot['geometry'].bounds['maxy'].item() - buf_n_m + gdf_shft_n_m
        return ul_x_utm, ul_y_utm

    def _crop_many_grid(self, plot_id_ul, pix_e_ul, pix_n_ul,
                       crop_e_m=9.170, crop_n_m=3.049,
                       crop_e_pix=None, crop_n_pix=None,
                       buf_e_pix=None, buf_n_pix=None,
                       buf_e_m=None, buf_n_m=None,
                       alley_size_e_m=None, alley_size_n_m=None,
                       alley_size_e_pix=None, alley_size_n_pix=None,
                       n_plots_x=5, n_plots_y=9, spyfile=None):
        '''
        Crops many plots from a single image by calculating the distance
        between plots based on ``crop_X_Y``, ``n_plots_X/Y``, and ``alley_size_X_Y``
        parameters.

        Parameters:
            plot_id_ul (``int``): the plot ID of the upper left (northwest-most)
                plot to be cropped.
            pix_e_ul (``int``): upper left pixel column (easting) of
                ``plot_id_ul``.
            pix_n_ul (``int``): upper left pixel row (northing) of ``plot_id_ul``.
            crop_e_m (``float``, optional): length of each row (easting
                direction) in the cropped image (in map units; e.g., meters).
            crop_n_m (``float``, optional): length of each column (northing
                direction) in the cropped image (in map units; e.g., meters).
            crop_e_pix (``int``, optional): number of pixels in each row in the
                cropped image.
            crop_n_pix (``int``, optional): number of pixels in each column in
                the cropped image.
            buf_e_m (``float``, optional): The buffer distance in the easting
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after ``crop_X_m`` / ``crop_X_pix``. A positive value will reduce the
                size of ``crop_X_m`` / ``crop_X_pix``, and a negative value will
                increase it.
            buf_n_m (``float``, optional): The buffer distance in the northing
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after ``crop_X_m`` / ``crop_X_pix``. A positive value will reduce the
                size of ``crop_X_m`` / ``crop_X_pix``, and a negative value will
                increase it.
            buf_e_pix (``int``, optional): The buffer distance in the easting
                direction (in pixel units) to be applied after calculating the
                original crop area.
            buf_n_pix (``int``, optional): The buffer distance in the northing
                direction (in pixel units) to be applied after calculating the
                original crop area.
            alley_size_e_m (``int``, optional): Should be passed if there are
                alleys passing across the E/W direction of the plots that are
                not accounted for by the ``crop_X_Y`` parameters. Used together
                with ``n_plots_x`` to determine the plots represented in a
                particular datacube.
            alley_size_n_m (``int``, optional): Should be passed if there are
                alleys passing across the N/S direction of the plots that are
                not accounted for by the ``crop_X_Y`` parameters. Used together
                with ``n_plots_y`` to determine the plots represented in a
                particular datacube.
            alley_size_e_pix (``float``, optional): see ``alley_size_e_m``.
            alley_size_n_pix (``float``, optional): see ``alley_size_n_m``.
            n_plots_x (``int``): number of plots in a row in east/west direction
                (default: 5).
            n_plots_y (``int``): number of plots in a row in north/south
                direction (default: 9).
            spyfile (``SpyFile`` object): The datacube to crop; if ``None``, loads
                datacube and band information from ``spatial_mod.spyfile``
                (default: ``None``).

        Returns:
            ``pandas.DataFrame``:
                - **df_plots** (``pandas.DataFrame``) -- data for
                  which to crop each plot; includes 'plot_id_ref', 'pix_e_ul', and
                  'pix_n_ul' columns. This data can be passed to
                  ``spatial_mod.crop_single()`` to perform the actual cropping.

        Note:
            Either the pixel coordinate or the map unit coordinate should be
            passed for ``crop_X_Y`` and ``buf_X_Y`` in each direction (i.e.,
            easting and northing). Do not pass both.

        Example:

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

#            crop_e_m (``float``, optional): length of each row (easting
#                direction) of the cropped image in map units (e.g., meters;
#                default: ``None``).
#            crop_n_m (``float``, optional): length of each column (northing
#                direction) of the cropped image in map units (e.g., meters;
#                default: ``None``)
#            crop_e_pix (``int``, optional): number of pixels in each row in the
#                cropped image (default: ``None``).
#            crop_n_pix (``int``, optional): number of pixels in each column in
#                the cropped image (default: ``None``).
#            buf_e_m (``float``, optional): The buffer distance in the easting
#                direction (in map units; e.g., meters) to be applied after
#                calculating the original crop area; the buffer is considered
#                after ``crop_X_m`` / ``crop_X_pix``. A positive value will
#                reduce the size of ``crop_X_m`` / ``crop_X_pix``, and a
#                negative value will increase it (default: ``None``).
#            buf_n_m (``float``, optional): The buffer distance in the northing
#                direction (in map units; e.g., meters) to be applied after
#                calculating the original crop area; the buffer is considered
#                after ``crop_X_m`` / ``crop_X_pix``. A positive value will
#                reduce the size of ``crop_X_m`` / ``crop_X_pix``, and a
#                negative value will increase it (default: ``None``).
#            buf_e_pix (``int``, optional): The buffer distance in the easting
#                direction (in pixel units) to be applied after calculating the
#                original crop area (default: ``None``).
#            buf_n_pix (``int``, optional): The buffer distance in the northing
#                direction (in pixel units) to be applied after calculating the
#                original crop area (default: ``None``).



    # crop_many_gdf should have crop_ and buf_ because otherwise it gets complicated
    # if you have to worry about all that during crop_single. In batch, these values
    # are passed in the spreadsheet, but if they are ignored during crop_many, then
    # there are a bunch of if/else statements deciding if the output of crop_many should
    # be used or if the spreadsheet should override the crop_many df..

    # In batch mode, it's easier to pass them directly to crop_many from the spreadsheet,
    # then let the df dictate everything that is passed to crop_single. We just have
    # to be sure that things like buf aren't passed twice (once in crop_many, then
    # again in crop_single), where the buffer might be applied twice.

    def _check_crop_defaults(self, **kwargs):
        '''
        Checks passed parameters against self.defaults.crop_defaults. The
        passed parameters takes precedence. If ``None`` is passed for a kwarg,
        defaults.crop_defaults is checked and the variable is populated from
        there.

        Parameters:
            **kwargs: A dict of keyword arguments to be checked against
                self.defaults.crop_defaults.
        '''
        for k in self.defaults.crop_defaults:  # all the allowed params
            # if k not in kwargs and self.defaults.crop_defaults[k] is not None:
            if k not in kwargs:
                # set to that of defaults, even if it is null so it is guarenteed to exist
                kwargs[k] = self.defaults.crop_defaults[k]
            elif kwargs[k] is None:  #  and self.defaults.crop_defaults[k] is not None:
                # set to that of defaults if passed as None
                kwargs[k] = self.defaults.crop_defaults[k]
        return kwargs

        # kwargs_null = {k:v for k,v in kwargs.items() if v is None}
        # for k in kwargs_null:
        #     if k in self.defaults.crop_defaults and self.defaults.crop_defaults[k] is not None:
        #         kwargs_null[k] = self.defaults.crop_defaults[k]
        # return kwargs_null

    def _append_null_rows_cols(self, array_crop, pix_e_ul_correct,
                               pix_n_ul_correct):
        '''
        Appends null rows or columns to the left or top of array_crop.

        This function is necessary when gdf spans to the west or south(?) of
        the image extend. In this case, spyfile.read_subregion() returns an
        invalid array.
        '''
        if pix_e_ul_correct is not None:
            # add abs(pix_e_ul_correct) nan columns to left of array_crop
            a = array_crop.copy()
            array_crop = np.zeros((
                a.shape[0],
                a.shape[1]+np.abs(pix_e_ul_correct),
                a.shape[2]), dtype=a.dtype)
            # array_crop[:] = np.nan
            array_crop[:,np.abs(pix_e_ul_correct):,:] = a

        if pix_n_ul_correct is not None:
            # add abs(pix_n_ul_correct) nan columns to top of array_crop
            a = array_crop.copy()
            array_crop = np.zeros((
                a.shape[0]+np.abs(pix_n_ul_correct),
                a.shape[1],
                a.shape[2]), dtype=a.dtype)
            # array_crop[:] = np.nan
            array_crop[np.abs(pix_n_ul_correct):,:,:] = a
        return array_crop

    def crop_many_gdf(self, spyfile=None, gdf=None, **kwargs):
            # plot_id_ref=None,
            # pix_e_ul=None, pix_n_ul=None,
            # crop_e_m=None, crop_n_m=None, crop_e_pix=None, crop_n_pix=None,
            # buf_e_m=None, buf_n_m=None, buf_e_pix=None, buf_n_pix=None,
            # gdf_shft_e_m=None, gdf_shft_n_m=None,
            # gdf_shft_e_pix=None, gdf_shft_n_pix=None, n_plots=None):
        '''
        Crops many plots from a single image by comparing the image to a
        polygon file (``geopandas.GoeDataFrame``) that contains plot
        information and geometry of plot boundaries.

        Parameters:
            spyfile (``SpyFile`` object, optional): The datacube to crop; if
                ``None``, loads datacube and band information from
                ``spatial_mod.spyfile`` (default: ``None``).
            gdf (``geopandas.GeoDataFrame``, optional): the plot IDs and
                polygon geometery of each of the plots; 'plot_id' must be used as
                a column name to identify each of the plots, and should be an
                integer; if ``None``, loads geodataframe from
                ``spatial_mod.gdf`` (default: ``None``).
            **kwargs: Can be any of the keys in self.defaults.crop_defaults:
                plot_id_ref (``int``, optional): the plot ID of the reference plot.
                    ``plot_id_ref`` is required if passing ``pix_e_ul``,
                    ``pix_n_ul``, or ``n_plots`` because it is used as the
                    reference point for any of the adjustments/modifications
                    dictated by said parameters. ``plot_id_ref`` must be present in
                    the ``gdf``, and the extent of ``plot_id_ref`` must intersect
                    the extent of the datacube (default: ``None``).
                pix_e_ul (``int``, optional): upper left pixel column (easting) of
                    ``plot_id_ref``; this is used to calculate the offset between
                    the GeoDataFrame geometry and the approximate image
                    georeference error (default: ``None``).
                pix_n_ul (``int``, optional): upper left pixel row (northing) of
                    ``plot_id_ref``; this is used to calculate the offset between
                    the GeoDataFrame geometry and the approximate image
                    georeference error (default: ``None``).
                crop_e_m (``float``, optional): length of each row (easting
                    direction) of the cropped image in map units (e.g., meters;
                    default: ``None``).
                crop_n_m (``float``, optional): length of each column (northing
                    direction) of the cropped image in map units (e.g., meters;
                    default: ``None``)
                crop_e_pix (``int``, optional): number of pixels in each row in the
                    cropped image (default: ``None``).
                crop_n_pix (``int``, optional): number of pixels in each column in
                    the cropped image (default: ``None``).
                buf_e_m (``float``, optional): The buffer distance in the easting
                    direction (in map units; e.g., meters) to be applied after
                    calculating the original crop area; the buffer is considered
                    after ``crop_X_m`` / ``crop_X_pix``. A positive value will
                    reduce the size of ``crop_X_m`` / ``crop_X_pix``, and a
                    negative value will increase it (default: ``None``).
                buf_n_m (``float``, optional): The buffer distance in the northing
                    direction (in map units; e.g., meters) to be applied after
                    calculating the original crop area; the buffer is considered
                    after ``crop_X_m`` / ``crop_X_pix``. A positive value will
                    reduce the size of ``crop_X_m`` / ``crop_X_pix``, and a
                    negative value will increase it (default: ``None``).
                buf_e_pix (``int``, optional): The buffer distance in the easting
                    direction (in pixel units) to be applied after calculating the
                    original crop area (default: ``None``).
                buf_n_pix (``int``, optional): The buffer distance in the northing
                    direction (in pixel units) to be applied after calculating the
                    original crop area (default: ``None``).
                gdf_shft_e_m (``float``): The distance to shift the cropped
                    datacube from the upper left/NW plot corner in the east
                    direction (negative value will shift to the west). Only
                    relevent when ``gdf`` is passed. This shift is applied after
                    the offset is applied from buf_X (default: 0.0).
                gdf_shft_n_m (``float``): The distance to shift the cropped
                    datacube from the upper left/NW plot corner in the north
                    direction (negative value will shift to the south). Only
                    relevent when ``gdf`` is passed. This shift is applied after
                    the offset is applied from buf_X (default: 0.0).
                gdf_shft_e_pix (``int``): The pixel units to shift the cropped
                    datacube from the upper left/NW plot corner in the east
                    direction (negative value will shift to the west). Only
                    relevent when ``gdf`` is passed. This shift is applied after
                    the offset is applied from buf_X (default: 0.0).
                gdf_shft_n_pix (``int``): The pixel units to shift the cropped
                    datacube from the upper left/NW plot corner in the north
                    direction (negative value will shift to the south). Only
                    relevent when ``gdf`` is passed. This shift is applied after
                    the offset is applied from buf_X (default: 0.0).
                n_plots (``int``, optional): number of plots to crop, starting with
                    ``plot_id_ref`` and moving from West to East and North to
                    South. This can be used to limit the number of cropped plots
                    (default; ``None``).

        Returns:
            ``pandas.DataFrame``:
                - **df_plots** (``pandas.DataFrame``) -- data for
                  which to crop each plot; includes 'plot_id_ref', 'pix_e_ul', and
                  'pix_n_ul' columns. This data can be passed to
                  ``spatial_mod.crop_single`` to perform the actual cropping.

        Note:
            If ``pix_e_ul`` or ``pix_n_ul`` are passed, the pixel offset from
            the northwest corner of ``plot_id_ref`` will be calculated. This
            offset is then applied to all plots within the extent of the image
            to systematically shift the actual upper left pixel locations for
            each plot, effectively shifting the easting and/or northing
            of the upper left pixel of the hyperspectral datacube to match that
            of the ``gdf``. If the shift should only apply to a select number
            of plots, ``n_plots`` can be passed to restrict the number of plots
            that are processed.

        Note:
            Either the pixel coordinate or the map unit coordinate should be
            passed for ``crop_X_Y`` and ``buf_X_Y`` in each direction (i.e.,
            easting and northing). Do not pass both.

        Example:
            Load the ``hsio`` and ``spatial_mod`` modules

            >>> import geopandas as gpd
            >>> import os
            >>> from hs_process import hsio
            >>> from hs_process import spatial_mod

            Read datacube and spatial plot boundaries

            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> fname_gdf = r'F:\\nigo0024\Documents\hs_process_demo\plot_bounds_small\plot_bounds.shp'
            >>> gdf = gpd.read_file(fname_gdf)
            >>> io = hsio(fname_in)
            >>> my_spatial_mod = spatial_mod(io.spyfile)
            >>> dir_out = os.path.join(io.base_dir, 'spatial_mod', 'crop_many_gdf')
            >>> name_append = '-crop-many-gdf'

            Get instructions on how plots should be cropped via
            ``spatial_mod.crop_many_gdf()``; note that a ``pandas.DataFrame``
            is returned with information describing how each plot should be
            cropped.

            >>> df_plots = my_spatial_mod.crop_many_gdf(spyfile=io.spyfile, gdf=gdf)
            >>> df_plots.head(5)
                plot_id_ref  pix_e_ul  pix_n_ul  crop_e_pix  crop_n_pix
            0          1018       478         0         229          76
            1           918       707         0         229          76
            2           818       936         0         229          76
            3           718      1165         0         229          76
            4           618      1394         0         229          76
            ...

            Use the data from the first row of df_plots to crop a single plot
            from the original image (uses spatial_mod.crop_single)

            >>> pix_e_ul=113
            >>> pix_n_ul=0
            >>> crop_e_pix=229
            >>> crop_n_pix=75
            >>> plot_id_ref=1018
            >>> array_crop, metadata = my_spatial_mod.crop_single(
                    pix_e_ul=pix_e_ul, pix_n_ul=pix_n_ul, crop_e_pix=crop_e_pix, crop_n_pix=crop_n_pix,
                    spyfile=io.spyfile, plot_id_ref=plot_id_ref)

            Save the cropped datacube and geotiff to a new directory

            >>> fname_out = os.path.join(dir_out, io.name_short + '_plot_' + str(1018) + name_append + '.' + io.defaults.envi_write.interleave)
            >>> fname_out_tif = os.path.join(dir_out, io.name_short + '_plot_' + str(1018) + '.tif')
            >>> io.write_cube(fname_out, array_crop, metadata=metadata, force=True)
            >>> io.write_tif(fname_out_tif, spyfile=array_crop, metadata=metadata)
            Saving F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1018-crop-many-gdf.bip
            Either `projection_out` is `None` or `geotransform_out` is `None` (or both are). Retrieving projection and geotransform information by loading `hsio.fname_in` via GDAL. Be sure this is appropriate for the data you are trying to write.

            Using a for loop, use ``spatial_mod.crop_single`` and
            ``hsio.write_cube`` to crop by plot and save cropped datacubes to
            file

            >>> for idx, row in df_plots.iterrows():
            >>>     io.read_cube(fname_in, name_long=io.name_long,
                                 name_plot=row['plot_id_ref'],
                                 name_short=io.name_short)
            >>>     my_spatial_mod.load_spyfile(io.spyfile)
            >>>     array_crop, metadata = my_spatial_mod.crop_single(
                            pix_e_ul=row['pix_e_ul'], pix_n_ul=row['pix_n_ul'],
                            crop_e_pix=row['crop_e_pix'], crop_n_pix=row['crop_n_pix'],
                            buf_e_m=2.0, buf_n_m=0.75,
                            plot_id_ref=row['plot_id_ref'])
            >>>     fname_out = os.path.join(dir_out, io.name_short + '_plot_' + str(row['plot_id_ref']) + name_append + '.bip.hdr')
            >>>     fname_out_tif = os.path.join(dir_out, io.name_short + '_plot_' + str(row['plot_id_ref']) + '.tif')
            >>>     io.write_cube(fname_out, array_crop, metadata=metadata, force=True)  # force=True to overwrite the plot_1018 image
            >>>     io.write_tif(fname_out_tif, spyfile=array_crop, metadata=metadata)
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1018.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_918.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_818.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_718.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_618.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_1017.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_917.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_817.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_717.bip
            Saving F:\\nigo0024\Documents\hs_process_demo\crop_many_gdf\Wells_rep2_20180628_16h56m_pika_gige_7_plot_617.bip
            ...

            Open cropped geotiff images in QGIS to visualize the extent of the
            cropped images compared to the original datacube and the plot
            boundaries (the full extent image is darkened and displayed in the
            background:

            .. image:: ../img/spatial_mod/crop_many_gdf_qgis.png
        '''
        kwargs_d = self._check_crop_defaults(**kwargs)

        if isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
        metadata = self.spyfile.metadata
        if gdf is None:
            gdf = self.gdf
        plot_id_ref = kwargs_d['plot_id_ref']
        msg1 = ('Please load a GeoDataFrame (geopandas library).\n')
        msg2 = ('Be sure "plot_id" is used as the column heading to identify '
                'plots in the GeodataFrame (`gdf`).\nGeoDataFrame (`gdf`) '
                'column names: {0}\n'.format(list(gdf.columns)))
        msg3 = ('Please be sure `plot_id` is present in `gdf` (i.e., '
                'the GeoDataFrame) and that plots are identified as integers.'
                '\nCurrent value of `plot_id_ref`: {0}\nGeoDataFrame '
                '(`gdf`) Plot ID data type: {1}\n'
                ''.format(plot_id_ref, type(gdf['plot_id'].loc[0])))
        assert isinstance(gdf, gpd.GeoDataFrame), msg1
        assert 'plot_id' in list(gdf.columns), msg2
        if pd.notnull(plot_id_ref):
            if plot_id_ref not in gdf['plot_id'].tolist():
                assert int(plot_id_ref) in gdf['plot_id'].tolist(), msg3
                plot_id_ref = int(plot_id_ref)
            else:
                assert plot_id_ref in gdf['plot_id'].tolist(), msg3
        df_plots = self._find_plots_gdf(gdf, plot_id_ref,
                                        kwargs_d['pix_e_ul'], kwargs_d['pix_n_ul'],
                                        kwargs_d['n_plots'], metadata)

        # if crop_X or buf_X were passed, overwrite them now
        crop_e_m, crop_n_m, crop_e_pix, crop_n_pix = self._pix_to_mapunit(
                kwargs_d['crop_e_m'], kwargs_d['crop_n_m'],
                kwargs_d['crop_e_pix'], kwargs_d['crop_n_pix'])
        buf_e_m, buf_n_m, buf_e_pix, buf_n_pix = self._pix_to_mapunit(
                kwargs_d['buf_e_m'], kwargs_d['buf_n_m'],
                kwargs_d['buf_e_pix'], kwargs_d['buf_n_pix'])
        gdf_shft_e_m, gdf_shft_n_m, gdf_shft_e_pix, gdf_shft_n_pix = self._pix_to_mapunit(
                kwargs_d['gdf_shft_e_m'], kwargs_d['gdf_shft_n_m'],
                kwargs_d['gdf_shft_e_pix'], kwargs_d['gdf_shft_n_pix'])
        if pd.notnull(crop_e_pix):
            df_plots['crop_e_pix'] = crop_e_pix
        if pd.notnull(crop_n_pix):
            df_plots['crop_n_pix'] = crop_n_pix
        if pd.notnull(buf_e_pix):
            df_plots['buf_e_pix'] = buf_e_pix
        if pd.notnull(buf_n_pix):
            df_plots['buf_n_pix'] = buf_n_pix
        if pd.notnull(gdf_shft_e_pix):
            df_plots['gdf_shft_e_pix'] = gdf_shft_e_pix
        if pd.notnull(gdf_shft_n_pix):
            df_plots['gdf_shft_n_pix'] = gdf_shft_n_pix
        return df_plots

    def crop_single(self, pix_e_ul=0, pix_n_ul=0, crop_e_pix=None,
                    crop_n_pix=None, crop_e_m=None, crop_n_m=None,
                    buf_e_pix=None, buf_n_pix=None, buf_e_m=None, buf_n_m=None,
                    spyfile=None, plot_id_ref=None, gdf=None,
                    gdf_shft_e_pix=None, gdf_shft_n_pix=None,
                    gdf_shft_e_m=None, gdf_shft_n_m=None,
                    name_append='spatial-crop-single'):
        '''
        Crops a single plot from an image. If ``plot_id_ref`` and ``gdf`` are
        explicitly passed (i.e., they will not be loaded from ``spatial_mod``
        class), the "map info" tag in the metadata will be adjusted to center
        the cropped area within the appropriate plot geometry.

        Parameters:
            pix_e_ul (``int``, optional): upper left pixel column (easting) to
                begin cropping (default: 0).
            pix_n_ul (``int``, optional): upper left pixel row (northing) to
                begin cropping (default: 0).
            crop_e_m (``float``, optional): length of each row (easting
                direction) of the cropped image in map units (e.g., meters;
                default: ``None``).
            crop_n_m (``float``, optional): length of each column (northing
                direction) of the cropped image in map units (e.g., meters;
                default: ``None``)
            crop_e_pix (``int``, optional): number of pixels in each row in the
                cropped image (default: ``None``).
            crop_n_pix (``int``, optional): number of pixels in each column in
                the cropped image (default: ``None``).
            buf_e_m (``float``, optional): The buffer distance in the easting
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after ``crop_X_m`` / ``crop_X_pix``. A positive value will
                reduce the size of ``crop_X_m`` / ``crop_X_pix``, and a
                negative value will increase it (default: ``None``).
            buf_n_m (``float``, optional): The buffer distance in the northing
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after ``crop_X_m`` / ``crop_X_pix``. A positive value will
                reduce the size of ``crop_X_m`` / ``crop_X_pix``, and a
                negative value will increase it (default: ``None``).
            buf_e_pix (``int``, optional): The buffer distance in the easting
                direction (in pixel units) to be applied after calculating the
                original crop area (default: ``None``).
            buf_n_pix (``int``, optional): The buffer distance in the northing
                direction (in pixel units) to be applied after calculating the
                original crop area (default: ``None``).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube to
                crop; if ``numpy.ndarray`` or ``None``, loads band information from
                ``self.spyfile`` (default: ``None``).
            plot_id_ref (``int``): the plot ID of the area to be cropped
                (default: ``None``).
            gdf (``geopandas.GeoDataFrame``): the plot names and polygon
                geometery of each of the plots; 'plot_id' must be used as a column
                name to identify each of the plots, and should be an integer.
            gdf_shft_e_m (``float``): The distance to shift the cropped
                datacube from the upper left/NW plot corner in the east
                direction (negative value will shift to the west). Only
                relevent when ``gdf`` is passed. This shift is applied after
                the offset is applied from buf_X (default: 0.0).
            gdf_shft_n_m (``float``): The distance to shift the cropped
                datacube from the upper left/NW plot corner in the north
                direction (negative value will shift to the south). Only
                relevent when ``gdf`` is passed. This shift is applied after
                the offset is applied from buf_X (default: 0.0).
            gdf_shft_e_pix (``int``): The pixel units to shift the cropped
                datacube from the upper left/NW plot corner in the east
                direction (negative value will shift to the west). Only
                relevent when ``gdf`` is passed. This shift is applied after
                the offset is applied from buf_X (default: 0.0).
            gdf_shft_n_pix (``int``): The pixel units to shift the cropped
                datacube from the upper left/NW plot corner in the north
                direction (negative value will shift to the south). Only
                relevent when ``gdf`` is passed. This shift is applied after
                the offset is applied from buf_X (default: 0.0).
            name_append (``str``): NOT YET SUPPORTED; name to append to the
                filename (default: 'spatial-crop-single').

        Returns:
            2-element ``tuple`` containing

            - **array_crop** (``numpy.ndarray``): Cropped datacube.
            - **metadata** (``dict``): Modified metadata describing the cropped
              hyperspectral datacube (``array_crop``).

        Example:
            Load and initialize the ``hsio`` and ``spatial_mod`` modules

            >>> from hs_process import hsio
            >>> from hs_process import spatial_mod
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)
            >>> my_spatial_mod = spatial_mod(io.spyfile)

            Crop an area with a width (easting) *200 pixels* and a height
            (northing) of *50 pixels*, with a northwest/upper left origin at
            the *342nd column* (easting) and *75th row* (northing).

            >>> pix_e_ul = 342
            >>> pix_n_ul = 75
            >>> array_crop, metadata = my_spatial_mod.crop_single(pix_e_ul, pix_n_ul, crop_e_pix=200, crop_n_pix=50)

            Save as a geotiff using ``io.write_tif``, then load into QGIS to
            visualize.

            >>> fname_tif = r'F:\\nigo0024\Documents\hs_process_demo\spatial_mod\crop_single\crop_single.tif'
            >>> io.write_tif(fname_tif, array_crop, metadata=metadata)
            Either `projection_out` is `None` or `geotransform_out` is `None` (or both are). Retrieving projection and geotransform information by loading `hsio.fname_in` via GDAL. Be sure this is appropriate for the data you are trying to write.

            Open cropped geotiff image in QGIS to visualize the extent of the
            cropped image compared to the original datacube and the plot
            boundaries (the full extent image is darkened and displayed in the
            background):

            .. image:: ../img/spatial_mod/crop_single_qgis.png
        '''
        crop_e_pix, crop_n_pix, crop_e_m, crop_n_m = self._handle_defaults(
                crop_e_pix, crop_n_pix, crop_e_m, crop_n_m, group='crop')
        buf_e_pix, buf_n_pix, buf_e_m, buf_n_m = self._handle_defaults(
                buf_e_pix, buf_n_pix, buf_e_m, buf_n_m, group='buffer')
        gdf_shft_e_pix, gdf_shft_n_pix, gdf_shft_e_m, gdf_shft_n_m = self._handle_defaults(
                gdf_shft_e_pix, gdf_shft_n_pix, gdf_shft_e_m, gdf_shft_n_m, group='gdf_shft')
        pix_e_ul, pix_e_lr = self._get_corners(pix_e_ul, crop_e_pix,
                                               buf_e_pix)
        pix_n_ul, pix_n_lr = self._get_corners(pix_n_ul, crop_n_pix,
                                               buf_n_pix)

        # read_subregion may return invalid array if gdf is of image
        # extent, so we have to have another way to get the cropped array.
        # pix_e_ul_correct = None
        # pix_n_ul_correct = None
        # if pix_e_ul < 0:
        #     pix_e_ul_correct = pix_e_ul
        #     pix_e_ul = 0
        # if pix_n_ul < 0:
        #     pix_n_ul_correct = pix_n_ul
        #     pix_n_ul = 0

        # read_subregion may return invalid array if gdf is outside image
        # extent, so we have to crop at zero and shift geotransform accordingly.
        if pix_e_ul < 0:
            if pd.isnull(gdf_shft_e_m):
                gdf_shft_e_m = 0
            gdf_shft_e_m += np.abs(pix_e_ul) * self.spy_ps_e
            pix_e_ul = 0
        if pix_n_ul < 0:
            if pd.isnull(gdf_shft_n_m):
                gdf_shft_n_m = 0
            gdf_shft_n_m += np.abs(pix_n_ul) * self.spy_ps_n
            pix_e_ul = 0

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

        # array_crop = self._append_null_rows_cols(
        #     array_crop, pix_e_ul_correct, pix_n_ul_correct)

        metadata = self.tools.spyfile.metadata
        map_info_set = metadata['map info']
        if isinstance(gdf, gpd.GeoDataFrame) and plot_id_ref is not None:
            msg1 = ('Please be sure `plot_id` is present in `gdf` (i.e., '
                    'the GeoDataFrame) and that plots are identified as '
                    'integers. \nCurrent value of `plot_id_ref`: {0}\n'
                    'GeoDataFrame (`gdf`) Plot ID data type: {1}\n'
                    ''.format(plot_id_ref, type(gdf['plot_id'].loc[0])))
            assert plot_id_ref in gdf['plot_id'].tolist(), msg1
            ul_x_utm, ul_y_utm = self._shift_by_gdf(gdf, plot_id_ref,
                                                    buf_e_m, buf_n_m,
                                                    gdf_shft_e_m, gdf_shft_n_m)
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
        Loads a ``SpyFile`` (Spectral Python object) for data access and/or
        manipulation by the ``hstools`` class.

        Parameters:
            spyfile (``SpyFile`` object): The datacube being accessed and/or
                manipulated.

        Example:
            Load and initialize the ``hsio`` and ``spatial_mod`` modules

            >>> from hs_process import hsio
            >>> from hs_process import spatial_mod
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)
            >>> my_spatial_mod = spatial_mod(io.spyfile)

            Load datacube using ``spatial_mod.load_spyfile``

            >>> my_spatial_mod.load_spyfile(io.spyfile)
            >>> my_spatial_mod.spyfile
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip'
        	# Rows:            617
        	# Samples:        1300
        	# Bands:           240
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32
        '''
        self.spyfile = spyfile
        self.tools = hstools(spyfile)
        try:
            map_info_set = self.tools.get_meta_set(self.spyfile.metadata['map info'])
            self.spy_ul_e_srs = float(map_info_set[3])
            self.spy_ul_n_srs = float(map_info_set[4])
            self.spy_ps_e = float(map_info_set[5])
            self.spy_ps_n = float(map_info_set[6])
        except KeyError as e:
            print('Map information was not able to be loaded from the '
                  '`SpyFile`. Please be sure the metadata contains the "map '
                  'info" tag with accurate geometric information.\n')
            self.spy_ul_e_srs = None
            self.spy_ul_n_srs = None
            self.spy_ps_e = None
            self.spy_ps_n = None
