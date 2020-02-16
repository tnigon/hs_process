# Welcome to hs_process
*An open-source Python package for geospatial processing of aerial hyperspectral imagery*

``hs_process`` **emphasizes the ability to batch process datacubes**, with the overall goal of keeping the processing pipeline as "hands-off" as possible. There is also a focus on maintaining the ability to have control over the subjective aspects of image processing (e.g., segmentation thresholds) and achieving repeatability in image post-processing.

## About
``hs_process`` is a Python package for geospatial post-processing of aerial hyperspectral imagery. The principle motivation for developing ``hs_process`` was to streamline the post-processing steps required prior to hyperspectral data analysis. Although commercial software perhaps exists for such needs, open source software that is both flexible and streamlined did not previously exist. The specific motivations were:

1. Minimize the time and effort required to perform post-processing tasks.
2. Minimize (and potentially eliminate) the number of steps where user intervention is required.
3. Provide a framework for post-processing an entire aerial hyperspectral dataset that, when repeated, does not require any user intervention.
4. Enable users to more easily document the complex processes and inherently subjective parameters used for post-processing aerial hyperspectral imagery (e.g., segmentation thresholds, spatial plot buffers, etc.).

In addressing these motivations, it was important to keep each post-processing steps as flexible as possible so the specific requirements of individual research objectives can be met.

For more information and tutorials, check out the [hs_process documentation](https://hs_process.readthedocs.io/en/latest/).

## Data requirements
The minimum data requirement to utilize this package is a pre-processed hyperspectral datacube (i.e., radiometric calibration, georectification, and reflectance conversion should have already been completed). Sample imagery captured from a [Resonon](https://resonon.com/) Pika II VIS-NIR line scanning imager can be downloaded from this [link](https://drive.google.com/drive/folders/1KpOBB4-qghedVFd8ukQngXNwUit8PFy_?usp=sharing>).

To perform spatial cropping, a polygon boundary file is required with "plot" column indicating the plot number of each boundary feature (an example geojson can be downloaded from this [link](https://drive.google.com/open?id=1fb1i46g88BcrTau0bwnWMrnDXo7FPH0p).

## Sample data
[Download from this link](https://drive.google.com/drive/folders/1KpOBB4-qghedVFd8ukQngXNwUit8PFy_?usp=sharing)

- ``hs_process`` was developed using images from the Resonon Pika II hyperspectral imager, so it tends to be tailored towards modifying metadata specific to Resonon, with the Pika II model imager in particular.
- ``hs_process`` leverages the [Spectral Python library](https://www.spectralpython.net) for reading and writing hyperspectral datacubes.

## Intended audience
This package was built for those looking for a clean, streamlined solution for post-processing aerial hyperspectral imagery. This package is especially well-suited for R&D departments looking for a straightforward way to achieve repeatable post-processing results so data analysis can proceed more quickly and systematically.

## Troubleshooting
Please report any issues you encounter through the [Github issue tracker](https://github.com/tnigon/hs_process/issues).
