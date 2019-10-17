# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:41:15 2019

@author: nigo0024
"""

from hs_process import batch
#from hs_process import hstools
#from hs_process import hsio

base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-29_AERF-plot2'
hs = hsp(base_dir,search_exp='.bip', recurs_level=0)
fname = hs.fname_list[0]
io = hsio(fname)
io = hsio()
array = io.img_sp.load()

tools = hstools(io.img_sp)
meta_bands = tools.get_meta_bands()


meta_bands = io.meta_bands


# In[1: debugging spectral mean 2019-06-26_Wells-V6]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-26_Wells-V6\spec_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[2: spectral mean 2019-08-02_Crookston_Anderson]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-08-02_Crookston_Anderson\cube_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[1: debugging spatial crop]
import os
from hs_process.utilities import hsio
from hs_process.spatial_mod import spatial_mod

directory = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2018\2018-07-11_Wells\Wells_rep4'
filename = r'Wells_rep4_20180711_18h22m_pika_gige_10-Radiance Conversion-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip'

io = hsio()
io.read_cube(os.path.join(directory, filename))
array = io.spyfile.load()
fname_tif = os.path.join(directory, 'test', 'test5.tif')

map_info = io.spyfile.metadata['map info']
utm_x = float(map_info[3])
utm_y = float(map_info[4])
size_x = float(map_info[5])
size_y = float(map_info[6])

pix_e_ul = 200
pix_n_ul = 237
crop_e_pix = 300
crop_n_pix = 50
my_spat = spatial_mod(io.spyfile)
array_crop, metadata = my_spat.crop_single(pix_e_ul, pix_n_ul, crop_e_pix, crop_n_pix)
utm_x_new, utm_y_new = my_spat.tools.get_UTM(pix_e_ul, pix_n_ul, utm_x, utm_y, size_x, size_y)

geotransform_out = [utm_x_new, 0.04, 0.0, utm_y_new, 0.0, -0.04]
io.write_tif(fname_tif, array_crop, geotransform_out=geotransform_out)


img_ds = io._read_envi_gdal(fname_in=os.path.join(directory, filename))
projection_out = img_ds.GetProjection()
geotransform_out = img_ds.GetGeoTransform()


crop_e_pix2, crop_n_pix2, crop_e_m2, crop_n_m2 = my_spat._handle_defaults(
                crop_e_pix, crop_n_pix, None, None)
my_spat._handle_defaults(None, None, None, None, group='buffer')

sw_e_m = plot_corners_m[0]
sw_n_m = plot_corners_m[1]
ne_e_m = plot_corners_m[2]
ne_n_m = plot_corners_m[3]

crop_e_pix = 90
crop_n_pix = 120
buffer_x_pix = 10
buffer_y_pix = 10

pix_e_lr = pix_e_ul + crop_e_pix
pix_n_lr = pix_n_ul + crop_n_pix
array_crop = io.spyfile.read_subregion((pix_n_ul, pix_n_lr), (pix_e_ul, pix_e_lr))

# In[ENVI_crop]
import os

from hs_process.utilities import hsio

directory = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2018\2018-07-11_Wells\Wells_rep4'
file = r'Wells_rep4_20180711_18h22m_pika_gige_10-Radiance Conversion-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip'

fname = os.path.join(directory, file)

hs = hsio(fname)

sm = spatial_mod(hs.spyfile)


plot_id_ul = 2008
df_plots = sm.envi_crop(plot_id_ul, pix_e_ul=233, pix_n_ul=60, plot_size_e_m=9.170,
             plot_size_n_m=3.049, alley_size_n_m=6.132,
             buf_x_m=1.5, buf_y_m=0.6, n_plots_x=5, n_plots_y=9)

# In[Batch crop Wells data]
import geopandas as gpd

from hs_process import batch
#from hs_process import spatial_mod

fname_sheet = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\python_processing\wells_image_cropping.csv'
fname_shp = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\GIS_files\plot_bounds\plot_bounds.shp'
gdf = gpd.read_file(fname_shp)

hsbatch = batch()
hsbatch.spatial_crop(fname_sheet,
                     folder_name='spatial_crop', name_append='spatial-crop',
                     geotiff=True, method='many_gdf', gdf=gdf, out_force=True)
# In[]
import pandas as pd
df_plots = pd.read_csv(fname_sheet)
for idx, row in df_plots.iterrows():
    print(row)

# In[]
import geopandas as gpd
import os

from hs_process.utilities import hsio
from hs_process.spatial_mod import spatial_mod

directory = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2018\2018-06-13_Wells\Wells_rep1'
fname = 'Wells_rep1_20180613_18h56m_pika_gige_1-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
hdr_name = os.path.join(directory, fname)
fname_shp = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\GIS_files\plot_bounds\plot_bounds.shp'

gdf = gpd.read_file(fname_shp)
hs = hsio(hdr_name)
my_spat_mod = spatial_mod(hs.spyfile, gdf)

plot_id_ref = 536
pix_e_ul = 524
pix_n_ul = 132
crop_e_m = 9.17
crop_n_m = 3.049
df_plots = my_spat_mod.crop_many(plot_id_ref, pix_e_ul, pix_n_ul, crop_e_m=crop_e_m,
                                 crop_n_m=crop_n_m)


df_plots2 = my_spat_mod.envi_crop(plot_id_ref, pix_e_ul, pix_n_ul, crop_e_m=crop_e_m,
                                  crop_n_m=crop_n_m, alley_size_n_m=6.132, buf_e_m=1.0, buf_n_m=0.5, n_plots_x=5, n_plots_y=9)
