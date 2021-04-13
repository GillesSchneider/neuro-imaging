# Haxby Registration 
The Python code registers [Haxby (1)](#references) to [MNI152 space (2/3)](#references) (~ 3H/subject).

## Why do we register Haxby to MNI space?

Because Haxby is not primarily aligned with MNI 152 space, it is impossible to apply parcellations on the dataset using atlases that are in MNI 152 space (e.g: [Yeo 2011 (3)](#references)). This way, registering Haxby to MNI 152 space leads to new ways to analyze the dataset.

## Pipeline

*  Get one volume from Haxby 
*  Initialize affine and rigid registrations
*  Load MNI152 template and resample it to Haxby
*  For every subject in Haxby:
   - For every 3D image:
     - Apply affine transformation 
     - Apply rigid transformation
     - Append the transformed image to a list
   - Save the transformed 4D image to folder "./save/subj"
*  Repeat

## Required Packages
_numpy, nibabel, nilearn, dipy, tqdm_


```sh
pip install numpy nibabel nilearn dipy tqdm
```

## Before Registration
Example of a Haxby 3D volume in the MNI152 template. The 3D volume is from subject 2 and is the mean along the axis 3, which is the time. 

<p align="center">
  <img  src="./images/before-registration.png">
</p>

The brain schematics is in MNI152 space in the above image. It can be seen that Haxby data is not aligned with MNI152 space.

```sh
#Code to print the above image
#Import nilearn
from nilearn import datasets 
from nilearn.image.image import mean_img
from nilearn.plotting import plot_glass_brain

#Second subject is choosen by default
haxby_dataset = datasets.fetch_haxby()

# Compute the mean EPI: we do the mean along the axis 3, which is time
func_filename = haxby_dataset.func[0]
mean_haxby = mean_img(func_filename)

plotting.plot_glass_brain(mean_haxby)

```
## Run code

```sh
python haxby-registration.py
```

## References

1. Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J., and Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science 293, 2425-2430.

2. VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins and BDCG, Unbiased average age-appropriate atlases for pediatric studies, NeuroImage, Volume 54, Issue 1, January 2011, ISSN 1053-8119, DOI: 10.1016/j.neuroimage.2010.07.033

3. VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins, Unbiased nonlinear average age-appropriate brain templates from birth to adulthood, NeuroImage, Volume 47, Supplement 1, July 2009, Page S102 Organization for Human Brain Mapping 2009 Annual Meeting, DOI: 10.1016/S1053-8119(09)70884-5

4. http://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011
