# IBC2 Pytorch Dataset for brain parcellation

The code creates a custom Pytorch Dataset for [Individual Brain Charting release 2 (1)](/https://doi.org/10.1038/s41597-020-00670-4) for brain parcellation. 
First of all, fetch IBC 2 with nilearn:

```sh
ibc_data = nilearn.datasets.fetch_neurovault_ids(collection_ids=["6618"])
```
Choose the subject(s), the tasks(s) and the constrast(s) within the task(s):

```sh
subjects = ['sub-02', 'sub-01']
tasks = ['hcp-motor'] #HCP Motor
contrasts =  ['Move left foot', 'Move left hand', 'Move right foot', 'Move tongue']
```

_Please refers to [Individual Brain Charting release 2 (1)](//https://doi.org/10.1038/s41597-020-00670-4) for more information about the data._

Get the dataset, and you are ready to go!

```sh
ibc_dataset = IBC2d(ibc_data, subjects, tasks, contrasts)
```
The dataset will contain 2D slices (z by default) of original images, corresponding 2D slices (z by default) after parcellation, and the contrast associated with them. 

## Disclaimer

Please note that the dataset works only with altas in MNI space. It will not work with parcellations such as Ward or Kmeans etc. For instance, use [BASC (2)](/http://dx.doi.org/10.1016/j.neuroimage.2010.02.082). 

## Packages

The following python packages are required: _numpy, sklearn, scipy, torch, nilearn, nibabel, itertools_.
Please make sure that all the packages mentioned above are installed prior to the use of the code.

## References

1. Pinho, A.L., Amadon, A., Gauthier, B. et al. Individual Brain Charting dataset extension, second release of high-resolution fMRI data for cognitive mapping. Sci Data 7, 353 (2020). https://doi.org/10.1038/s41597-020-00670-4

2. Bellec P, Rosa-Neto P, Lyttelton OC, Benali H, Evans AC, Jul. 2010. Multi-level bootstrap analysis of stable clusters in resting-state fMRI. NeuroImage 51 (3), 1126-1139. URL http://dx.doi.org/10.1016/j.neuroimage.2010.02.082
