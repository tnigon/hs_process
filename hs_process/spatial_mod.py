# -*- coding: utf-8 -*-
import json
import os

class Spatial_mod(object):
    '''
    Class for manipulating data within the spatial domain (e.g., cropping by
    geographical boundary)
    '''
    def __init__(self, img_sp):
        self.img_sp = img_sp

        self.buf_x_pix = None
        self.buf_y_pix = None
        self.geotransform = None
        self.pix_skip = int(6.132 / -0.04)  # alley - skip 6.132 m
        self.projection = None
        self.plot_cols = None
        self.plot_rows = None
        self.row_plots_top = 0
        self.row_plots_bot = 0
        self.size_x_m = None  # get from .hdr
        self.size_y_m = None
        self.ul_x_m = None
        self.ul_y_m = None

        def run_init():
            try:
                self.size_x_m = float(self.img_sp.metadata['map info'][5])
                self.size_y_m = float(self.img_sp.metadata['map info'][6])
                self.ul_x_m = float(self.img_sp.metadata['map info'][4])
                self.ul_y_m = float(self.img_sp.metadata['map info'][3])
            except KeyError as e:
                self.size_x_m = None
                self.size_y_m = None
                self.ul_x_m = None
                self.ul_y_m = None

        run_init()

    def _check_alley(self):
        '''
        Calculates whether there is an alleyway in the image (based on plot
        configuration), then adjusts plot_rows so it is correct after
        considering the alley
        '''
        plot_id_tens = abs(self.plot_id_ul) % 100
        self.row_plots_top = plot_id_tens % 9
        if self.row_plots_top == 0:
            self.row_plots_top = self.plot_rows  # remainder is 0, not 9..

        if self.row_plots_top < self.plot_rows:
            # get pix left over
            pix_remain = (self.img_sp.nrows - abs(self.ul_y_pix) -
                          (self.row_plots_top * abs(self.plot_y_pix)))
        else:
            return

        if pix_remain >= abs(self.pix_skip + self.plot_y_pix):
            # have room for more plots (must still remove 2 rows of plots)
            # calculate rows remain after skip
            self.row_plots_bot = int(abs((pix_remain + self.pix_skip) /
                                     self.plot_y_pix))
            self.plot_rows = self.row_plots_top + self.row_plots_bot
        elif pix_remain >= abs(self.plot_y_pix) * 2:
            # remove 2 rows of plots
            self.plot_rows -= 2
        elif pix_remain >= abs(self.plot_y_pix):
            # remove 1 row of plots
            self.plot_rows -= 1
        else:
            # works out perfect.. don't have to change anything
            pass

    def _get_UTM(self, ulx, uly, utm_x, utm_y, size_x=0.04, size_y=-0.04):
        '''
        Calculates the new UTM coordinate of cropped plot
        '''
        utm_x_new = utm_x + (ulx * size_x)
        utm_y_new = utm_y - (uly * size_y)
        return utm_x_new, utm_y_new

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

    def crop_many(self, base_dir_crop=None):
        '''
        Iterates through all plots, crops each, and saves to file
        '''
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
                    xsize=abs(self.plot_x_pix - (self.buf_x_pix * 2)),
                    ysize=abs(self.plot_y_pix - (self.buf_y_pix * 2)))


            array_img_crop = self.img_sp.read_subregion(
                    (abs(row_pix), abs(row_pix) + abs(self.plot_y_pix - (self.buf_y_pix * 2))),
                    (abs(col_pix), abs(col_pix) + abs(self.plot_x_pix - (self.buf_x_pix * 2))))
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
                                                   utm_y, size_x=self.size_x_m,
                                                   size_y=self.size_y_m)
            geotransform_out = [ul_x_utm, self.size_x_m, 0.0, ul_y_utm, 0.0,
                                self.size_y_m]
            self._write_envi(array_img_crop, fname_out_envi, geotransform_out)
            self._modify_hdr(fname_out_envi)
            fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'
            self._write_tif(array_img_crop, fname_out_tif, geotransform_out)

#    def _read_envi_hdr(fname_hdr):
#        '''
#        Reads ENVI .hdr file and
#        '''
#        with open(fname_hdr, 'r') as f:
#            data = f.readlines()
#        matches = []
#        regex1 = re.compile(r'^(.+?)\s*=\s*({\s*.*?\n*.*?})$',re.M|re.I)
#        regex2 = re.compile(r'^(.+?)\s*=\s*(.*?)$',re.M|re.I)
#        for line in data:
#            matches.extend(regex1.findall(line))
#            subhdr = regex1.sub('', line)  # remove from line
#            matches.extend(regex2.findall(subhdr))
#        return dict(matches)
#
    def crop_single_batch(self, fname_sheet, plot_x_pix=90, plot_y_pix=120,
                          interleave='bip', name_append='crop',
                          base_dir_out=None, plot_name=True):
        '''
        Iterates through spreadsheet that provides necessary information about
        how each image should be cropped and how it should be saved
        '''
        df_plots = pd.read_csv(fname_sheet)

        for idx, row in df_plots.iterrows():
            directory = row['directory']
            name_short = row['name_short']
            name_long = row['name_long']
            ext = row['ext']
            pix_e_ul = row['easting_pix']
            pix_n_ul = row['northing_pix']

            fname_in = os.path.join(directory, name_short+name_long+ext)
            self.crop_single(fname_in, pix_e_ul, pix_n_ul,
                             plot_x_pix=plot_x_pix, plot_y_pix=plot_y_pix,
                             interleave=interleave, name_long=name_long,
                             name_append=name_append,
                             base_dir_out=base_dir_out, plot_name=plot_name)

    def crop_single(self, fname_in, pix_e_ul, pix_n_ul, plot_x_pix=90,
                    plot_y_pix=120, interleave='bip',
                    name_long='-Unit Conversion Utility', base_dir_out=None,
                    name_append='crop', plot_name=True):
        '''
        Crops and saves an image

        Parameters:
            fname_in (`str`): hyperspectral image filename to be cropped
            pix_e_ul (`int`): upper left column (easting)to begin cropping
            pix_n_ul (`int`): upper left row (northing) to begin cropping
            plot_x_pix (`int`): number of pixels per row in the cropped image
            plot_y_pix (`int`): number of pixels per colum in the cropped image
            interleave (`str`): interleave (and file extension) of cropped
                image; (default: 'bip').
            name_long (str): Spectronon processing appends processing names to
                the filenames; this indicates those processing names that are
                repetitive and can be deleted from the filename following
                processing.
            base_dir_out (`str`): output directory of the cropped image
                (default: `None`)
            name_append (`str`): text to append to end of filename; used to
                describe this particular manipulation (default: 'crop').
            plot_name (bool): Indicates whether image (and its filename) is for
                an individual plot (True), or for many plots (False) (default:
                True).
        '''
        self.read_cube(fname_in, name_long=name_long, plot_name=plot_name)
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out=base_dir_out, folder_name='crop')

        pix_e_new = pix_e_ul + plot_x_pix
        pix_n_new = pix_n_ul + plot_y_pix
        array_img_crop = self.img_sp.read_subregion((pix_n_ul, pix_n_new),
                                                    (pix_e_ul, pix_e_new))

        if name_append is None:
            name_append = 'crop'
        name_label = (name_print + self._xstr(name_append) + '.' +
                      interleave)
        fname_out_envi = os.path.join(base_dir_out, name_label)
        print('Spatially cropping image: {0}'.format(name_print))

#        if self.name_plot is not None:
#            fname = (self.name_plot + self._xstr(name_append) + '.' +
#                     interleave)
#        else:
#            fname = (self.name_short + self._xstr(name_append) + '.' +
#                     interleave)
#        fname_out_envi = os.path.join(base_dir_out, fname)

#        print('Cropping plot {0}'.format(self.plot))

        map_info_set = self.metadata['map info']
        utm_x = self._get_meta_set(map_info_set, 3)
        utm_y = self._get_meta_set(map_info_set, 4)
        ul_x_utm, ul_y_utm = self._get_UTM(pix_e_ul, pix_n_ul, utm_x,
                                           utm_y, size_x=self.size_x_m,
                                           size_y=self.size_y_m)
        map_info_set = self._modify_meta_set(map_info_set, 3, ul_x_utm)
        map_info_set = self._modify_meta_set(map_info_set, 4, ul_y_utm)
        self.metadata['map info'] = map_info_set
        hist_str = (" -> Hyperspectral.crop_single[<"
                    "SpecPyFloatText label: 'pix_e_ul?' value:{0}; "
                    "SpecPyFloatText label: 'pix_n_ul?' value:{1} >]"
                    "".format(pix_e_ul, pix_n_ul))
        self.metadata['history'] += hist_str
        self.metadata['samples'] = array_img_crop.shape[1]
        self.metadata['lines'] = array_img_crop.shape[0]
        self.metadata['label'] = name_label

        envi.save_image(fname_out_envi + '.hdr', array_img_crop,
                        dtype=np.float32, force=True, ext=None,
                        interleave=interleave, metadata=self.metadata)

        fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'

#            rgb_list = [self._get_band(640)[0],
#                        self._get_band(550)[0],
#                        self._get_band(460)[0]]
#            from spectral import save_rgb
#            save_rgb(fname_out_tif, array_img_crop, rgb_list, format='tiff')

        img_ds = self._get_envi_gdal(fname_in=fname_in)
        projection_out = img_ds.GetProjection()
        img_ds = None  # I only want to use GDAL when I have to..

#        drv = gdal.GetDriverByName('ENVI')
#        drv.Register()
#        img_ds = gdal.Open(fname_in, gdalconst.GA_ReadOnly)
#        projection_out = img_ds.GetProjection()
#        img_ds = None
#        drv = None

        geotransform_out = [ul_x_utm, self.size_x_m, 0.0, ul_y_utm, 0.0,
                            self.size_y_m]
        self._write_tif(array_img_crop, fname_out_tif, projection_out,
                        geotransform_out)
