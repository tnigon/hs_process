# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression

from hs_process.utilities import defaults
from hs_process.utilities import hsio


#plt.style.use('ggplot')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('text', usetex=False)


class analyze(object):
    '''
    Class for exploratory data analysis on a hyperspectral dataset. Each
    analysis tool in this class may have different data requirements, but
    generally both the spectra and "ground truth" data to be predicted are
    required.
    '''
    def __init__(self, base_dir=None, search_ext='.bip', dir_level=0):
        '''
        Parameters:
            base_dir (`str`, optional): directory path to search for files to
                load in for anlysis; this is not required as each analysis
                tool in this class may have different data requirements.
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

    def _check_bands_wls(self, wl, b, n):
        '''
        Checks to be sure there is a valid wavelength or band to be used
        '''
        msg1 = ('At least one of either `wl{0}` or `b{0}` must be passed. '
                'Please check your inputs.\n'.format(n))
        if wl is None and b is None:
            raise ValueError(msg1)
        msg2 = ('Both `wl{0} and b{0} were passed, but only one can be used '
                'to deterimine which image band to use; the wavelength band '
                '(`wl{0}`={1}) will be used for the band math operation.\n'
                ''.format(n, wl))
        if wl is not None and b is not None:
            print(msg2)
        elif wl is not None and b is None:  # use wl
            pass
        elif wl is None and b is not None:  # get wl
            wl = self.io.tools.get_wavelength(b)
        return wl

    def _build_empty_arrays(self, meta_bands):
        '''
        Builds empty arrays for correlation matrices
        '''
        wl_list = [wl for wl in meta_bands.values()]
        df = pd.DataFrame(columns=['wl', 'd_wl', 'skip_n'])
        df['wl'] = wl_list
        df['d_wl'] = df['wl'] - df['wl'].shift(-1)
        dif_mode = df['d_wl'].mode()[0]
        df['skip_n'] = np.nan_to_num(np.round(list(df['d_wl'].shift(1))/dif_mode)-1).astype(int)
        array_size = len(meta_bands) + df['skip_n'].sum()
        array_results1 = np.zeros((array_size, array_size-1))
        array_results2 = array_results1.copy()
        array_results3 = array_results1.copy()
        return df, array_results1, array_results2, array_results3

    def _get_band_list(self, wl_list, list_range):
        '''
        Determines how a list of wavelengths should be consolidated, if at all.
        '''
        if isinstance(wl_list, list) and list_range is True:
            msg = ('When using a `list_range`, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            assert len(wl_list) == 2, msg
            band_list = self.io.tools.get_band_range(wl_list, index=False)
        elif isinstance(wl_list, list) and list_range is False:
            band_list = []
            for b_i in wl_list:
                b = self.io.tools.get_band(b_i)
                band_list.append(b)
        else:  # just a single band; disregards `list_range`
            b = self.io.tools.get_band(wl_list)
            band_list = [b]
        return band_list

    def _get_lambda_str(self, array, meta_bands):
        '''
        Gets wavelengths with highest R2 value
        '''
        idx_wl1, idx_wl2 = np.where(array == np.amax(array))
        # To find number of rows to skip, we have sum the rows of df until we
        # reach idx_wl1, and the number of rows that we've gone through is the
        # band number we're on
        df, _, _, _ = self._build_empty_arrays(meta_bands)
        band_1 = 0
        array_idx1 = 0

        for idx, row in enumerate(df.iterrows()):
            band_1 = idx + 1
            array_idx1 += int(row[1]['skip_n'] + 1)
            if array_idx1 >= idx_wl1[0]:
                break
        band_2 = 0
        array_idx2 = 0
        for idx, row in enumerate(df.iterrows()):
            band_2 = idx + 1
            array_idx2 += int(row[1]['skip_n'] + 1)
            if array_idx2 >= idx_wl2[0]:
                break
#        band_1 = idx_wl1[0] + 1
#        band_2 = idx_wl2[0] + 1
        wl1 = meta_bands[band_1]
        wl2 = meta_bands[band_2]
        r2_max1 = r'Max $R^{2}$: '
        r2_max2 = '{0:.3f}'.format(np.amax(array))
        lambda1_ = r'$\lambda 1 = ${0:.0f} nm'.format(wl1)
        lambda2_ = r'$\lambda 2 = ${0:.0f} nm'.format(wl2)
        lambda_str = (r2_max1 + r2_max2 + '\n' + lambda1_ + '\n' + lambda2_)
        return lambda_str, wl1, wl2

    def _get_wavelength_list(self, band_list_in, list_range):
        '''
        Determines how a list of bands should be consolidated, if at all.
        '''
        if isinstance(band_list_in, list) and list_range is True:
            msg = ('When using a `list_range`, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            assert len(band_list_in) == 2, msg
            wl_list = self.tools.get_band_range(band_list_in, index=False)
        elif isinstance(band_list_in, list) and list_range is False:
            wl_list = []
            for b_i in band_list_in:
                wl = self.tools.get_wavelength(b_i)
                wl_list.append(wl)
        else:  # just a single band; disregards `list_range`
            wl = self.tools.get_wavelength(band_list_in)
            wl_list = [wl]
        return wl_list

    def _plot_set_labels(self, ax, title_str, eq_str, date_str,
                         growth_stage_str, lambda_str, wl1, wl2,
                         color='#444444', fontsize=16):
        '''
        Sets labels
        '''
        ax.tick_params(labelsize=fontsize)
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
        el = mpatches.Ellipse((0, 0), width=0.3, height=0.3, angle=50,
                              alpha=0.5)
        ax.add_artist(el)

        if title_str is not None:
    #        title_str = r'NDI vs. $d$AONR at Preplant (kg ha$^{-1}$)'
            ax.set_title(title_str, color='#282828',
                         fontsize=int(fontsize * 1.1), fontweight='bold')
        if eq_str is not None:
            ax.annotate(
                eq_str,
                xy=(0.95, 0.1),
                xycoords=ax.transAxes,
                ha='right', va='bottom', fontsize=int(fontsize * 0.65),
                color=color,
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)))
        if lambda_str is not None:
            ax.annotate(
                lambda_str,
                xy=(wl2, wl1),
                xytext=(0.95, 0.25),  # loc to place text
                textcoords='axes fraction',  # placed relative to axes
                ha='right',  # alignment of text
                va='bottom',
                fontsize=int(fontsize * 0.7),
                color=color,
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5), alpha=0.7),
                arrowprops=dict(arrowstyle='-|>',
                                color=color,
                                patchB=el,
                                shrinkA=0,
                                shrinkB=0,
                                connectionstyle='arc3,rad=-0.3'))

        if date_str is not None:
            plt.text(1, 1.01, date_str, color=color,
                     fontsize=int(fontsize * 0.7),
                     horizontalalignment='right',
                     transform=ax.transAxes, fontweight='bold')
        if growth_stage_str is not None:
            plt.text(0, 1.01, growth_stage_str, color=color,
                     fontsize=int(fontsize * 0.7),
                     horizontalalignment='left',
                     transform=ax.transAxes, fontweight='bold')
        return ax

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
            out_files: include the full pathname, filename, and ext of all
                files that have `search_exp` in their name.
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


    def band_math_derivative(self, df, wl1=None, wl2=None, wl3=None,
                             b1=None, b2=None, b3=None, list_range=True,
                             normalized=False):

        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        wl3 = self._check_bands_wls(wl3, b3, 3)
        # Now, band input is converted to wavelength input and this can be used
        msg = ('A valid pandas.DataFrame must be passed.\n')
        assert isinstance(df, pd.DataFrame), msg

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        band3_list = self._get_band_list(wl3, list_range)
        s_b1 = self.io.tools.get_spectral_mean(band1_list, df)
        s_b2 = self.io.tools.get_spectral_mean(band2_list, df)
        s_b3 = self.io.tools.get_spectral_mean(band3_list, df)
        if normalized is True:
            _, wl1_mean = self.io.tools.get_center_wl(band1_list, wls=False)
            _, wl2_mean = self.io.tools.get_center_wl(band2_list, wls=False)
            _, wl3_mean = self.io.tools.get_center_wl(band3_list, wls=False)
            s_der = (((s_b1-s_b2) / (wl1_mean-wl2_mean)) /
                     ((s_b2-s_b3) / (wl2_mean-wl3_mean)))
        else:
            s_der = (s_b1-s_b2)/(s_b2-s_b3)

        str_name = 'der_{0}_{1}_{2}'.format(int(np.mean(band1_list)),
                                            int(np.mean(band2_list)),
                                            int(np.mean(band3_list)))
        s_der = s_der.rename(str_name)
        return s_der

    def band_math_ndi(self, df, wl1=None, wl2=None, b1=None, b2=None,
                      meta_bands=None, list_range=True):
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        # Now, band input is converted to wavelength input and this can be used
        msg = ('A valid pandas.DataFrame must be passed.\n')
        assert isinstance(df, pd.DataFrame), msg

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        s_b1 = self.io.tools.get_spectral_mean(band1_list, df)
        s_b2 = self.io.tools.get_spectral_mean(band2_list, df)
        s_ndi = (s_b1-s_b2)/(s_b1+s_b2)
        str_name = 'ndi_{0}_{1}'.format(int(np.mean(band1_list)),
                                        int(np.mean(band2_list)))
        s_ndi = s_ndi.rename(str_name)
        return s_ndi

    def band_math_ratio(self, df, wl1=None, wl2=None, b1=None, b2=None,
                        meta_bands=None, list_range=True):
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        # Now, band input is converted to wavelength input and this can be used
        msg = ('A valid pandas.DataFrame must be passed.\n')
        assert isinstance(df, pd.DataFrame), msg

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        s_b1 = self.io.tools.get_spectral_mean(band1_list, df)
        s_b2 = self.io.tools.get_spectral_mean(band2_list, df)
        s_ratio = s_b1/s_b2
        str_name = 'ratio_{0}_{1}'.format(int(np.mean(band1_list)),
                                          int(np.mean(band2_list)))
        s_ratio = s_ratio.rename(str_name)
        return s_ratio

    def band_math_mcari2(self, df, wl1=None, wl2=None, wl3=None, b1=None,
                         b2=None, b3=None, meta_bands=None, list_range=True):
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        wl3 = self._check_bands_wls(wl3, b3, 3)
        # Now, band input is converted to wavelength input and this can be used
        msg = ('A valid pandas.DataFrame must be passed.\n')
        assert isinstance(df, pd.DataFrame), msg

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        band3_list = self._get_band_list(wl3, list_range)
        s_b1 = self.io.tools.get_spectral_mean(band1_list, df)
        s_b2 = self.io.tools.get_spectral_mean(band2_list, df)
        s_b3 = self.io.tools.get_spectral_mean(band3_list, df)
        s_mcari2 = ((1.5 * (2.5 * (s_b1 - s_b2) - 1.3 * (s_b1 - s_b3))) /
                    np.sqrt((2 * s_b1 + 1)**2 - (6 * s_b1 - 5 * np.sqrt(s_b2)) - 0.5))
        str_name = 'mcari2_{0}_{1}_{2}'.format(int(np.mean(band1_list)),
                                               int(np.mean(band2_list)),
                                               int(np.mean(band3_list)))
        s_mcari2 = s_mcari2.rename(str_name)
        return s_mcari2

    def coefficient_matrix(self, df_ref, y1, y2=None, y3=None, method='ndi',
                           meta_bands=None):
        '''
        Calculates the coefficient between y1/y2 and the spectral index for
        every combination of bands in df_ref
        '''
        if meta_bands is None:
            meta_bands = self.io.tools.meta_bands
        bands = list(meta_bands.keys())

        df, array_results1, array_results2, array_results3 =\
            self._build_empty_arrays(meta_bands)

        for idx1, band1 in enumerate(reversed(bands)):
            # get all bands LOWER than band
            band_list2 = [i for i in reversed(bands) if i < band1]
            # get row to put result in
            skip_n_row = df.iloc[:band1-1, [2]].sum()[0]
            idx_row = band1 + skip_n_row - 1
            for idx2, band2 in enumerate(band_list2):
                # get column to put result in
                skip_n_col = df.iloc[:band2-1, [2]].sum()[0]
                idx_col = band2 + skip_n_col - 1
                if method == 'ndi':
                    s_ndi = self.band_math_ndi(df_ref, b1=band1, b2=band2,
                                               meta_bands=meta_bands,
                                               list_range=False)
                elif method == 'ratio':
                    s_ndi = self.band_math_ratio(df_ref, b1=band1, b2=band2,
                                                 meta_bands=meta_bands,
                                                 list_range=False)
#                mx, b, r_value, p_value, std_err = stats.linregress(s_ndi, y1)
                reg1 = LinearRegression().fit(s_ndi.values.reshape(-1,1),
                                              y1.values.reshape(-1,1))
                r_2 = reg1.score(s_ndi.values.reshape(-1,1),
                                 y1.values.reshape(-1,1))
                array_results1[idx_row, idx_col] = r_2
                if y2 is not None:
                    reg2 = LinearRegression().fit(s_ndi.values.reshape(-1,1),
                                                  y2.values.reshape(-1,1))
                    r_2 = reg2.score(s_ndi.values.reshape(-1,1),
                                     y2.values.reshape(-1,1))
                    array_results2[idx_row, idx_col] = r_2
                if y3 is not None:
                    reg3 = LinearRegression().fit(s_ndi.values.reshape(-1,1),
                                                  y3.values.reshape(-1,1))
                    r_2 = reg3.score(s_ndi.values.reshape(-1,1),
                                     y3.values.reshape(-1,1))
                    array_results3[idx_row, idx_col] = r_2
        if y3 is not None:
            array_results3 = np.ma.masked_where(array_results3 == 0,
                                                array_results3)
            array_results2 = np.ma.masked_where(array_results2 == 0,
                                                array_results2)
            array_results1 = np.ma.masked_where(array_results1 == 0,
                                                array_results1)
            return array_results1, array_results2, array_results3
        elif y2 is not None:
            array_results2 = np.ma.masked_where(array_results2 == 0,
                                                array_results2)
            array_results1 = np.ma.masked_where(array_results1 == 0,
                                                array_results1)
            return array_results1, array_results2
        else:
            array_results1 = np.ma.masked_where(array_results1 == 0,
                                                array_results1)
            return array_results1

    def plot_coefficient_matrix(self, array, meta_bands=None, contours=True,
                                title_str=None,
                                eq_str=None, date_str=None,
                                growth_stage_str=None, fname_out=None,
                                color='#282828', fontsize=16,
                                style='seaborn-whitegrid',
                                cmap='viridis'):
        '''
        Plots a coefficient matrix represented in `array`.

        Paramters:
            color (`str`): hex color for labels (default '#282828')
            fontsize (`int`): font size to use for axes labels; all other
                labels are scaled appropriately for display (default: 16)
            style (`str`): matplotlib style to use for plot (default:
                'seaborn-whitegrid')
            cmap (`str`): matplotlib color map to use for displaying the
                correlation matrix. (default: 'viridis')
        '''
        plt.style.use(style)
        if meta_bands is None:
            meta_bands = self.io.tools.meta_bands
        wls = list(meta_bands.values())

        g, ax1 = plt.subplots()
        img = ax1.imshow(array, cmap=cmap, origin='lower',
                         extent=[min(wls), max(wls), min(wls), max(wls)])
        cbar = g.colorbar(img)
        cbar.ax.set_ylabel(r'$R^{2}$', color=color, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        ax1.set_xlabel(r'$\lambda 2$ (nm)', color=color, fontsize=fontsize,
                       fontweight='bold')
        ax1.set_ylabel(r'$\lambda 1$ (nm)', color=color, fontsize=fontsize,
                       fontweight='bold')
        lambda_str, wl1, wl2 = self._get_lambda_str(array, meta_bands)
        ax1 = self._plot_set_labels(ax1, title_str, eq_str, date_str,
                                    growth_stage_str, lambda_str, wl1, wl2,
                                    fontsize=fontsize)
        if contours is True:
            x = np.linspace(min(wls), max(wls), array.shape[1])
            y = np.linspace(min(wls), max(wls), array.shape[0])
            X, Y = np.meshgrid(x, y)
            CS = ax1.contour(X, Y, array,
                             levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             alpha=0.7,
                             linewidths=0.7,
                             linestyles='dashed',
                             colors=color)
            ax1.clabel(CS, inline=1, fontsize=fontsize*0.5)


        plt.tight_layout()
        if fname_out is not None:
            if not os.path.isdir(os.path.dirname(fname_out)):
                try:
                    os.mkdir(os.path.dirname(fname_out))
                except FileNotFoundError:
                    os.mkdir(os.path.dirname(os.path.split(fname_out)[0]))
                    os.mkdir(os.path.dirname(fname_out))
            g.savefig(fname_out)
        return g, ax1

#        array_1_flip = np.flip(array_1, axis=0)
#        idx_wl1, idx_wl2 = np.where(array_1_flip == np.amax(array_1_flip))
#        band_1 = array_1.shape[0] - (idx_wl1[0])
#        band_2 = idx_wl2[0] + 1
#        wl1 = meta_bands[band_1]
#        wl2 = meta_bands[band_2]
#        r2_max1 = r'Max $R^{2}$: '
#        r2_max2 = '{0:.3f}'.format(np.amax(array_1))
#        lambda1_ = r'$\lambda 1 = ${0:.0f} nm'.format(wl1)
#        lambda2_ = r'$\lambda 2 = ${0:.0f} nm'.format(wl2)
#        lambda_str = (r2_max1 + r2_max2 + '\n' + lambda1_ + '\n' + lambda2_)
#        ax1.text(0.95, 0.4, lambda_str, fontsize=10,
#                 horizontalalignment='right', verticalalignment='center',
#                 transform=ax1.transAxes, fontweight='bold')

    def spectra_to_df(self, fname_list=None, base_dir=None, search_ext='spec',
                      dir_level=0):
        '''
        Reads all the .spec files in a direcory and returns their data as a
        pandas.DataFrame object.
        '''
        if fname_list is None and base_dir is not None:
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)
        elif fname_list is None and base_dir is None:
            # base_dir may have been stored to the `batch` object
            base_dir = self.base_dir
            msg = ('Please set `fname_list` or `base_dir` to indicate which '
                   'datacubes should be processed.\n')
            assert base_dir is not None, msg
            fname_list = self._recurs_dir(base_dir, search_ext, dir_level)

        # load the data from the Spectral Python (SpyFile) object
        df_spec = None
        for fname in fname_list:
            self.io.read_spec(fname + '.hdr')
            meta_bands = self.io.tools.meta_bands
            array = self.io.spyfile_spec.load()
            data = list(np.reshape(array, (array.shape[2])))
            data.insert(0, self.io.name_plot)
            data.insert(0, os.path.basename(fname))
            if df_spec is None:
                bands = list(meta_bands.keys())
                bands.insert(0, 'plot_id')
                bands.insert(0, 'fname')
                df_spec = pd.DataFrame(columns=bands)
            df_spec_temp = pd.DataFrame(data=[data], columns=bands)
            df_spec = df_spec.append(df_spec_temp)
        try:
            df_spec['plot_id'] = pd.to_numeric(df_spec['plot_id'])
        except ValueError:
            print('Unable to convert "plot_id" column to numeric type.\n')
        return df_spec


