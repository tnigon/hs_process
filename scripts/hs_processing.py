# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:45:09 2019

@author: nigo0024
"""

# In[1.a. Get mean spectra from multiple spectra]

base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-26_Wells-V6\spec_ref_panels'

my_hs = Hyperspectral()
fname_list = my_hs._recurs_dir(base_dir=base_dir_spec, search_exp='.spec')
df_specs = None

# In[1.b. Load spectra]
for fname_spec in fname_list:
    my_hs.read_spec(fname_spec)
    if df_specs is None:
        df_specs = pd.DataFrame(my_hs.img_sp.asarray()[0,0,:],
                                columns=[os.path.basename(fname_spec)],
                                dtype=float)
    else:
        df_temp = pd.DataFrame(my_hs.img_sp.asarray()[0,0,:],
                               columns=[os.path.basename(fname_spec)],
                               dtype=float)
        df_specs = df_specs.join(df_temp)

df_mean = df_specs.mean(axis=1)
df_mean = df_mean.rename('mean')
df_std = df_specs.std(axis=1)
df_std = df_std.rename('std')
df_cv = df_mean / df_std
df_cv = df_cv.rename('cv')

array_index = df_mean.to_numpy()
array_index = array_index.reshape(len(df_mean),1,1)

# In[1.c. Save to ENVI file]
fname_out = os.path.join(base_dir_spec, 'spec_mean_spy.spec')
my_hs._write_spec_spy(array_index, fname_out, interleave='bil')

#my_hs._write_spec(array_index, fname_out)
#my_hs._write_hdr(fname_out)

# In[1.d. Plot]
import seaborn as sns

sns.lineplot(x=range(240), y=array_index[:,0, 0])

# In[]

# In[2.a. Get spectra for each pixel in directory]
base_dir_spec = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-08-28_Waseca-AERF\cube_ref_panels'
my_hs = Hyperspectral()
fname_list = my_hs._recurs_dir(base_dir=base_dir_spec, search_exp='.bip')
df_specs = None

# In[2.b. Load image spectra]
for fname_spec in fname_list:
    my_hs.read_cube(fname_spec)
    array = my_hs.img_sp.asarray()
    pixels = array.reshape((array.shape[0]*array.shape[1]), array.shape[2])
    if df_specs is None:
#        df_specs = pd.DataFrame(my_hs.img_sp.asarray()[0,0,:],
#                                columns=[os.path.basename(fname_spec)],
#                                dtype=float)
        df_specs = pd.DataFrame(pixels, dtype=float)
    else:
        df_temp = pd.DataFrame(pixels, dtype=float)
#        df_temp = pd.DataFrame(my_hs.img_sp.asarray()[0,0,:],
#                               columns=[os.path.basename(fname_spec)],
#                               dtype=float)
        df_specs = df_specs.append(df_temp, ignore_index=True)

df_mean = df_specs.mean()
df_mean = df_mean.rename('mean')
df_std = df_specs.std()
df_std = df_std.rename('std')
df_cv = df_mean / df_std
df_cv = df_cv.rename('cv')

#array_mean = df_mean.to_numpy()
#array_mean = array_mean.reshape(len(df_mean),1,1)
#
#array_std = df_std.to_numpy()
#array_std = array_std.reshape(len(df_mean),1,1)

# In[2.c. Save to ENVI file]
fname_out = os.path.join(base_dir_spec, 'spec_mean_spy.spec')
my_hs._write_spec_spy(fname_out, df_mean, df_std, interleave='bip')

# In[]

# In[3.a. Calculate NDVI]

base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-26_Wells-V6\all_panels'

my_hs = Hyperspectral()
fname_list = my_hs._recurs_dir(base_dir=base_dir, search_exp='.bip', level=0)
df_ndi_stats = pd.DataFrame(columns=['fname', 'mean', 'std', 'cv'])
name_long = ('-Radiance From Raw Data-Georectify Airborne Datacube-'
             'Reflectance from Radiance Data and Measured Reference Spectrum')
for idx, fname_in in enumerate(fname_list):
    my_hs.read_cube(fname_in, spectra_smooth=True,
                    name_long=name_long)
    ndi_array = my_hs.band_math(b1=780, b2=559, form='ndi')
    mean = np.nanmean(ndi_array)
    std = np.nanstd(ndi_array)
    cv = std/mean
    df_ndi_temp = pd.DataFrame([[fname_in, mean, std, cv]],
                               columns=['fname', 'mean', 'std', 'cv'])
    df_ndi_stats = df_ndi_stats.append(df_ndi_temp, ignore_index=True)

df_ndi_stats.to_csv(r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-26_Wells-V6\closest_panel\all_panels\ndi_stats.csv')

# In[]

# In[4.a. Clip and smooth]
base_dir = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-29_AERF-plot2'

my_hs = Hyperspectral()
fname_list = my_hs._recurs_dir(base_dir=base_dir, search_exp='.bip', level=0)
df_smooth_stats = pd.DataFrame(columns=['fname', 'mean', 'std', 'cv'])
name_long = ('-Radiance From Raw Data-Georectify Airborne Datacube-'
             'Reflectance from Radiance Data and Measured Reference Spectrum')

for idx, fname_in in enumerate(fname_list):
    my_hs.read_cube(fname_in, name_long=name_long)
    my_hs.spectral_clip_and_smooth(base_dir_out=None, name_append=None,
                                   wl_bands=[[0, 420], [760, 776], [813, 827],
                                             [880, 1000]],
                                   spectra_smooth=True, window_size=11, order=2,
                                   save_out=True, interleave='bip')
    mean = np.nanmean(my_hs.array_smooth)
    std = np.nanstd(my_hs.array_smooth)
    cv = std/mean
    df_smooth_temp = pd.DataFrame([[fname_in, mean, std, cv]],
                                  columns=['fname', 'mean', 'std', 'cv'])
    df_smooth_stats = df_smooth_stats.append(df_smooth_temp, ignore_index=True)

# In[5.a. Crop plot2 by the stakes]

fname_sheet = r'G:\BBE\AGROBOT\Shared Work\Data\PikaImagery4_Reflectance\2019\2019-06-29_AERF-plot2\smooth_spec_clip\crop\aerf_plot2_stakes.csv'

my_hs = Hyperspectral()
my_hs.crop_single_batch(fname_sheet, plot_x_pix=90, plot_y_pix=120,
                        interleave='bip', name_append='crop-stakes',
                        base_dir_out=None, plot_name=True)











