# HS_process
A Python package for processing and manipulating aerial hyperspectral imagery.

## Introduction
*hs_process* emphasizes the ability to batch process datacubes, with the overall goal of keeping the processing pipeline as "hands-off" as possible. There is also a focus of maintaining the ability to have control over the subjective aspects of processing.

## Notes
- *hs_process* was developed using imager from the Resonon Pika II hyperspectral imager, so it tends to be tailored towards modifying metadata specific to Resonon, with the Pika II in particular.
- *hs_process* leverages the [Spectral Python library](https://www.spectralpython.net) for reading and writing hyperspectral datacubes.
