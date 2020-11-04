Release: 0.0.4
***************
**Date**: 2020

**Description**: Fixes to the batch.spatial_crop function

* Added ``gdf_shft_e_m`` and ``gdf_shft_n_m`` to ``spatial_mod.crop_single`` function. This adds flexibility for shifting a crop plot in some direction so it is better centered within the ``gdf`` plot boundary.
* Changed column name requirement from "plot" to "plot_id" to indicate plot ID in ``gdf`` (relevant for ``spatial_mod.crop_single`` and ``spatial_mod.crop_many_gdf`` functions).
* When cropping by ``gdf`` bounds in ``spatial_mod``, if ``gdf`` bounds are outside image extent, the cropped image had an extra "buffer" equal to the length between the ``gdf`` bound and corresponding extent of the image. Fixed this unintended behavior so the cropped image has the expected extent based on crop_XXX, buf_XXX, etc.
* Renamed ``spatial_mod.crop_single`` ``plot_id`` parameter to ``plot_id_ref`` to be consistent with ``spatial_mod.crop_many_gdf``.
* Many bug fixes making the spatial_crop function more robust to various scenarios and to perform quality control checks on input geodataframe of plot boundaries for spatial cropping.
* During batch processing, if out_force is ``False`` and files exist in output directory, hs_process will skip over files instead of raising an error that the file already exists. This is useful if many files have already been processed and you'd like to process the remaining files without reprocessing all those that are completed.
* Added an option to create a mask between two thresholds or percentiles (applies to ``hstools.mask_array`` and ``batch.segment_create_mask``).
* Added options to ``batch.segment_create_mask`` to provide a choice whether the datacube and .spec files are saved to file. This potentially saves disk space if the datacube isn't needed in subsequent analyses.
* User can optionally pass a lists for ``mask_dir``, ``mask_side``, and ``mask_thresh``/``mask_percentile`` to ``batch.segment_create_mask``
* Added a progress bar the following batch functions: ``cube_to_spectra``, ``segment_composite_band``, ``segment_band_math``, ``segment_create_mask``, ``spectral_clip``, and ``spectral_smooth``.
* Added ``get_wavelength_range`` to ``utilities`` class.
* Added ``spec_mod.spectral_mimic`` and ``spec_mod.spectral_resample`` functions.

Release: 0.0.3
***************
**Date**: 2020 February 17

**Description**: Trying to build so hs_process works with Anaconda builds.

* Adjusted the requirements to be only geopandas, seaborn, and spectral (others are dependencies of these three).
* data, tests, and example folders added to installation
* shortened name of sample datacube because pip won't copy it with such a long filename

Release: 0.0.2
***************
**Date**: 2020 February 17

**Description**: Uploading a source distribution to PyPI

* Simplified the tests to use a small datacube

Release: 0.0.1
***************
**Date**: 2020 February 16

**Description**: Initial release