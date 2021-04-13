#!/usr/bin/env python
# coding: utf-8




# # Haxby (~10 hours)
import os
import numpy as np
np.set_printoptions(precision=4,suppress = True)
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import datasets
import dipy
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D)
from nilearn.datasets import load_mni152_template
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
import time
from nilearn.input_data import NiftiMasker
import nilearn
from tqdm import tqdm
from nilearn import plotting
from nilearn import masking
subjects = (1,2,3,4,5,6)
haxby = datasets.fetch_haxby(subjects=subjects)
anat = nb.load(haxby.anat[0])
bold = nb.load(haxby.func[0])

volume = bold.get_fdata()[:,:,:,0]
volume = nb.Nifti1Image(volume,bold.affine)

if os.path.exists('./save'):
    pass
else:
    os.mkdir('./save')

#Affine registration
nbins = 32
sampling_prop = None
level_iters = [1000, 100, 10]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

metric = MutualInformationMetric(nbins, sampling_prop)
affreg = AffineRegistration(metric=metric, level_iters = level_iters, sigmas = sigmas, factors = factors)
transform = TranslationTransform3D()
params0 = None

#Non linear registration
level_iters = [10, 10, 5]

metric = CCMetric(3)
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

#Registration on Haxby subjects

##Load MNI template
template_img = load_mni152_template()
template_mask = nilearn.datasets.load_mni152_brain_mask()
template_img = nilearn.image.resample_to_img(template_img,volume)

template_affine = template_img.affine
template_data = template_img.get_fdata()
subj = 0

for subject in haxby.func:
    
    subj += 1
    ##Load images
    moving_imgs = nb.load(subject)
    
    ##Apply mask
    masker = NiftiMasker(mask_img=haxby.mask, memory='nilearn_cache')
    moving_imgs = masker.fit_transform(moving_imgs)
    moving_imgs = masker.inverse_transform(moving_imgs)
    
    moving_affine = moving_imgs.affine
    moving_header = moving_imgs.header
    moving_data = moving_imgs.get_fdata()
    
    print('Images of subject ' + str(subj) + ' loaded')
    
    ##List of the volumes - Type : NiftiImage
    moving_volumes = []
    
    print('subject {} : computing registration...'.format(subj))
    
    for vol_idx in tqdm(range(moving_data.shape[3]),desc='subject {}'.format(subj)):
        start_time = time.process_time()
        
        volume = moving_data[:,:,:,vol_idx]
        
        ##Compute affine registration
        affine = affreg.optimize(template_data, volume, transform, params0, template_affine, moving_affine)
        transformed = affine.transform(volume)
        
        ##Compute rigid registration
        mapping = sdr.optimize(template_data, volume, template_affine, moving_affine, affine.affine)
        transformed = mapping.transform(volume)
                
        nifti_img = nb.Nifti1Image(transformed,template_affine)
        
        moving_volumes.append(nifti_img)
        
        print("computation time (1 volume)", round(time.process_time() - start_time), "seconds")
        
       
    print('subject {} : registration done !'.format(subj))    
   
     
    ##Concat volumes, then save
    transformed_imgs = nilearn.image.concat_imgs(moving_volumes)
    
    
    path = './save/subj{}/'.format(subj)
    
    if os.path.exists(path):
        nb.save(transformed_imgs, path + 'bold_registred.nii.gz')
    else:
        os.mkdir(path)
        nb.save(transformed_imgs, path + 'bold_registred.nii.gz')
    
    print('subject {} : data saved'.format(subj))

