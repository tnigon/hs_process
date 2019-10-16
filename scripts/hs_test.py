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
import geopandas as gpd
from hs_process.utilities import hsio

fname_shp = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\GIS_files\plot_bounds\plot_bounds.shp'

gdf = gpd.read_file(fname_shp)

'''
If we know the plot and the pixel of the upper left corner of the
northwest-most plot, we can use the plot_bounds shapefile to look up the
geometry/coordinates of every other plot in that image, then apply the same
offset to each of those plots as was applied based on the calculated offset of
the northwest-most plot. For example:
'''
directory = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2018\2018-07-11_Wells\Wells_rep4'
filename = r'Wells_rep4_20180711_18h22m_pika_gige_10-Radiance Conversion-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip'
plot_id = 2008
pix_e_ul = 233
pix_n_ul = 60
io = hsio()

io.read_cube(os.path.join(directory, filename))
array = io.spyfile.load()

# get southwest and northwest corner (in map units)
plot_corners_m = gdf[gdf['plot'] == plot_id]['geometry'].total_bounds

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

if buffer_x_pix is not None:
    pix_e_ul += buffer_x_pix
    pix_e_lr -= buffer_x_pix
if buffer_y_pix is not None:
    # TODO: check that this is applied in correct direction
    pix_n_ul += buffer_y_pix
    pix_n_lr -= buffer_y_pix

array_crop_buf = io.spyfile.read_subregion((pix_n_ul, pix_n_lr), (pix_e_ul, pix_e_lr))

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
from hs_process import batch
from hs_process import spatial_mod

fname_sheet = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\python_processing\wells_image_cropping.csv'

hsbatch = batch()
hsbatch.spatial_crop(fname_sheet,
                     folder_name='spatial_crop', name_append='spatial-crop',
                     geotiff=True, method='many')
# In[]
import pandas as pd
df_plots = pd.read_csv(fname_sheet)









