# pylossmaps

This library facilitates the fetching and handling of the LHC's BLM measurements.

# Installation

tba

# Requierements

* [pytimber](https://www.github.com/rdemaria/pytimber)
* pandas
* numpy
* matplotlib

Optional:
* tqdm

# Usage

**See examples in the `notebooks` folder.**

There are 3 main classes:

## The BLMDataFetcher class:

This class will handle the fetching of BLM data and assigning the correct header information.

If has 2 main methods of fetching data:
    * `from_datetime`: which takes 2 datetime objects or epoch/unix time numbers.
    * `from_fill`: which takes a fill number along with the requeted beam modes.

It also has a helper function to return data surronding triggers of the ADT blowup:
    * `iter_from_adt`: iteratively yield BLMData instances of data surronding the trigger of the ADT within the requested time range.

In order to facilitate the fetching of BLM background for data following an ADT trigger:
    * `bg_from_ADT_trigger`: provided the datetime of the ADT trigger, it will figure out a time interval where the no ADT triggers occur and fetch the background data.

The fetcher class returns the data in the shape of a a BLMData instance.

## The BLMData class:

This class handles the BLM data, the main methods are:
    * `plot`: creates a waterfall plot of the BLM data.
    * `iter_max`: iterates on index of the max values of the desired BLMS, defaults to the primary IR 7 BLMs.
    * `get_beam_meta`: fetches to get some additional information on the beam, i.e. intensity, number of bunches, fillign scheme, ..., for the current time range.

You can access the raw data through the `data` attribute.

In order to create a loss map of a given time, use the `loss_map` method, and provide the desired datetime. This outputs a LossMap instance.

## The LossMap class:

 This class handles the loss map processing. It provides a pandas/numpy style interface for filtering and selecting BLMs.
 Some main methods are:
    * `plot`: to create a loss map plot.
    * `set_background`: to set another LossMap instance as the background signal.
    * `normalize`: to normalize the data to the max value, or the signal of the provided BLM.
    * various methods for filtering based on `cell` number, `IR`, `side`, collimator type, `beam`, ...
