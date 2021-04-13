# Haxby Registration
> Register Haxby Data to MNI template

## Required Packages
_numpy, nibabel, nilearn, dipy, tqdm_


```sh
pip install numpy nibabel nilearn dipy tqdm
```

## Before Registration
> Haxby image in the MNI152 template.

<p align="center">
  <img  src="./images/before-registration.png">
</p>

Code to print the above image:
```sh
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

