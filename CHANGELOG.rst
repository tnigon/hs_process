Release: 0.0.4
***************
**Date**: 2020

**Description**: Fixes to the batch.spatial_crop function

* Added ``gdf_shft_e_m`` and ``gdf_shft_n_m`` to ``spatial_mod.crop_single`` function. This adds flexibility for shifting a crop plot in some direction so it is better centered within the ``gdf`` plot boundary.
* Many bug fixes making the spatial_crop function more robust to various scenarios.
* During batch processing, if out_force is ``False`` and files exist in output directory, hs_process will skip over files instead of raising an error that the file already exists. This is useful if many files have already been processed and you'd like to process the remaining files without reprocessing all those that are completed.
* Added an option to create a mask between two thresholds or percentiles (applies to ``hstools.mask_array`` and ``batch.segment_create_mask``).
* Added options to ``batch.segment_create_mask`` to provide a choice whether the datacube and .spec files are saved to file. This potentially saves disk space if the datacube isn't needed in subsequent analyses.
* User can optionally pass a lists for ``mask_dir``, ``mask_side``, and ``mask_thresh``/``mask_percentile`` to ``batch.segment_create_mask``
* Added a progress bar the following batch functions: ``cube_to_spectra``, ``segment_composite_band``, ``segment_band_math``, ``segment_create_mask``, ``spectral_clip``, and ``spectral_smooth``.

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