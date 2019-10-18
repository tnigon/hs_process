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


# In[1a: spectral mean 2019-06-26_Wells-V6]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-26_Wells-V6\spec_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[1b: spectral mean 2019-08-02_Crookston_Anderson]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-08-02_Crookston_Anderson\cube_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[1c: spectral mean 2019-08-28_AERF-plot2]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-08-28_AERF-plot2\cube_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[1d: spectral mean 2019-07-08_LTARN]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-07-08_LTARN\cube_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[1e: spectral mean 2019-07-22_Waseca-LTARN]
from hs_process import batch

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-07-22_Waseca-LTARN\cube_ref_panels'
hsbatch = batch(base_dir_spec, search_ext='.bip')
hsbatch.spectra_combine(base_dir=base_dir_spec, search_ext='bip', dir_level=0,
                        out_force=True)

# In[2. Batch crop - Wells data]
import geopandas as gpd

from hs_process import batch

fname_sheet = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\python_processing\wells_image_cropping.csv'
fname_shp = r'G:\BBE\AGROBOT\Shared Work\Wells_Study\GIS_files\plot_bounds\plot_bounds.shp'
gdf = gpd.read_file(fname_shp)

hsbatch = batch()

method = 'many_gdf'  # options: 'single', 'many_grid', or 'many_gdf'
hsbatch.spatial_crop(fname_sheet,
                     folder_name='spatial_crop_{0}'.format(method),
                     name_append='spatial-crop', geotiff=True, method=method,
                     gdf=gdf, out_force=True)

# In[3. Segment troubleshooting]
import os

from hs_process.utilities import hsio
from hs_process.segment import segment

directory = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2018\2018-06-13_Wells\Wells_rep1'
fname = 'Wells_rep1_20180613_18h56m_pika_gige_1-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
hdr_name = os.path.join(directory, fname)

hs = hsio(hdr_name)
my_seg = segment(hs.spyfile)

array_ndi, metadata = my_seg.band_math_ndi(b1=780, b2=559, list_range=True)

array_ndi_r, metadata = my_seg.band_math_ndi(b1=[760, 810], b2=[530, 570], list_range=True)
array_ndi_nr, metadata_nr = my_seg.band_math_ndi(b1=[760, 770, 780, 810], b2=[530, 550, 570], list_range=False)

array_ndi_nr2, metadata_nr = my_seg.band_math_ndi(b1=[760, 770, 780, 810], b2=550, list_range=False)



# In[5. 67th percentile, segment, calculate plot avg for all bands, and export to spreadsheet]

# In[5.a. Calculate NDVI]

base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2018\2018-06-13_Wells\crop_3_standardized_byreflectance\block1\conventional\smooth_spec_clip'

my_hs = Hyperspectral()
fname_list = my_hs._recurs_dir(base_dir=base_dir, search_exp='.bip', level=0)
df_ndi_stats = pd.DataFrame(columns=['fname', 'gndvi_whole', 'gndvi_67th_pctl',
                                     'std_67th_pctl', 'cv_67th_pctl'])
columns = None
df_hs_veg_mean = None
df_hs_veg_std = None
for idx, fname_in in enumerate(fname_list):
    my_hs.read_cube(fname_in, name_long=None)
    if columns is None:
        columns = list(my_hs.meta_bands.values())
        columns.insert(0, 'fname')
        df_hs_veg_mean = pd.DataFrame(columns=columns)
        df_hs_veg_std = pd.DataFrame(columns=columns)

    ndi_array = my_hs.band_math_ndi(b1=760, b2=720)
    pctl_67 = np.percentile(ndi_array, 90)
    ndi_array_ma = np.ma.masked_where(ndi_array<pctl_67, ndi_array)

    array_hs = my_hs.img_sp.load()
    ndi_array_ma_all_bands = np.dstack([ndi_array_ma]*array_hs.shape[2])
    array_hs_ma = np.ma.masked_where(ndi_array_ma_all_bands<pctl_67, array_hs)
    mean_hs = list(np.nanmean(array_hs_ma, axis=(0,1)))
    std_hs = list(np.nanstd(array_hs_ma, axis=(0,1)))
    mean_hs.insert(0, my_hs.name_short)
    std_hs.insert(0, my_hs.name_short)
    df_hs_veg_mean_temp = pd.DataFrame([mean_hs], columns=columns)
    df_hs_veg_std_temp = pd.DataFrame([std_hs], columns=columns)

    df_hs_veg_mean = df_hs_veg_mean.append(df_hs_veg_mean_temp, ignore_index=True)
    df_hs_veg_std = df_hs_veg_std.append(df_hs_veg_std_temp, ignore_index=True)

    # Calculate mean of non-masked GNDVI
    mean = np.nanmean(ndi_array)
    mean_ma = np.nanmean(ndi_array_ma)
    std = np.nanstd(ndi_array_ma)
    cv = std/mean_ma
    df_ndi_temp = pd.DataFrame([[my_hs.name_short, mean, mean_ma, std, cv]],
                               columns=['fname', 'gndvi_whole',
                                        'gndvi_67th_pctl', 'std_67th_pctl',
                                        'cv_67th_pctl'])
    df_ndi_stats = df_ndi_stats.append(df_ndi_temp, ignore_index=True)

df_hs_veg_mean.to_csv(os.path.join(my_hs.base_dir_out, 'hs_veg_mean_90_pctl.csv'),
                    index=False)
df_hs_veg_std.to_csv(os.path.join(my_hs.base_dir_out, 'hs_veg_std_90_pctl.csv'),
                    index=False)
df_ndi_stats.to_csv(os.path.join(my_hs.base_dir_out, 'ndi_stats_90_pctl.csv'),
                    index=False)






