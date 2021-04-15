import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

import os
import shutil
import random
import json 

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from scipy.optimize import fsolve

import torch
from torch.utils.data import DataLoader,random_split
from torch.utils.data.dataset import Dataset
from torch.utils.data import SubsetRandomSampler, Subset

import nilearn
import nibabel as nb
from nilearn import datasets
from nilearn.input_data import NiftiMasker,NiftiLabelsMasker

'''
...IBC 2 dataset for brain parcellation:
...Works for IBC release 2 AND only with atlas (ex: BASC 64, 122, etc.) in the MNI space
...The dataset contains 2D slices (z) and is contrast specific within task
'''
#ibc_data = nilearn.datasets.fetch_neurovault_ids(collection_ids=["6618"])

class IBC():
	#This class aims to search images of subject(s)/task(s)/condition(s)
	
	def __init__(self, ibc_data):
		#Get the lists of subject names, contrasts, tasks and images
				
		self.subjectnames = [cur['name'][:6] for cur in ibc_data.images_meta]
		self.contrasts = [cur['contrast_definition'] for cur in ibc_data.images_meta]
		self.tasks = [cur['task'] for cur in ibc_data.images_meta]
		self.images = ibc_data.images

	def __getsourcetarget__(self, target_tasks:list, target_subjects:list, target_contrasts:list):

		#Find the corresponding tasks
		bin_tasks = find(self.tasks, target_tasks) #binarize
		subjectnames = list(compress(self.subjectnames, bin_tasks))
		allcontrasts = list(compress(self.contrasts, bin_tasks))
		alltasks = list(compress(self.tasks, bin_tasks))
		images = list(compress(self.images, bin_tasks)) 

		#Select subjects
		bin_subjectnames = find(subjectnames, target_subjects)
		subjectnames = list(compress(subjectnames, bin_subjectnames))
		allcontrasts = list(compress(allcontrasts, bin_subjectnames))
		alltasks = list(compress(alltasks, bin_subjectnames))
		images = list(compress(images, bin_subjectnames))
		
		#Select contrasts, make sure that contrasts are valid
		bin_contrasts = find(allcontrasts, target_contrasts)
		allcontrasts = list(compress(allcontrasts, bin_contrasts))
		subjectnames = list(compress(subjectnames, bin_contrasts))
		images = list(compress(images, bin_contrasts))

		return images, allcontrasts, alltasks


class IBC2d(Dataset):
	
	def __init__(self, ibc_data:dict, subjects:list, tasks:list, contrasts:list, parcel:NiftiLabelsMasker, axis=2):
		#Create IBC 2 dataset under specific constraints
		#ibc_data : dictionnary fetched with nilearn
		#subjects : list of subject(s)
		#tasks : list of task(s)
		#contrast : list of constrast(s)
		#parcel: the desired parcellation
		#axis : the axis used to cut volumes, default : 2 -> z

		#Search the images that fit the conditions
		ibc = IBC(ibc_data)
		source_fns, contrasts, tasks = ibc.__getsourcetarget__(tasks, subjects, contrasts)
		
		#Encode contrasts to integers
		label_encoder = preprocessing.LabelEncoder()
		label_encoder.fit(contrasts)

		self.axis = axis
		self.parcel = parcel
		self.src_arrs = source_fns
		self.ibc = ibc
		self.tasks = tasks
		
		imgs, init = [], nb.load(source_fns[0])
		
		#Stack the images
		#For each path image
		for img in source_fns:
			#Load the image
			img = nb.load(img)
			#Get image data and add it to a list
			imgs.append(img.get_fdata(dtype=np.float32))
		
		#Concatenate the images to get a single 4D image
		src_arrs = np.stack(imgs, axis=3)
		
		#Compute parcellation
		nib_src = nb.Nifti1Image(src_arrs, init.affine)
		nib_tgt = parcel.fit_transform(nib_src)
		nib_tgt = parcel.inverse_transform(nib_tgt)
		
		#Create a folder to save parcellations
		#TO-DO: ask the path to user
		path = './parc/'
		if os.path.exists(path):
			pass
		else:
			os.mkdir(path)
		
		tgt_fns = []

		for idx in range(nib_tgt.shape[3]):
			tgt_fns.append('{}/{}.nii.gz'.format(path, idx))
			nb.save(nib_tgt[:,:,:,idx],'{}/{}.nii.gz'.format(path, idx))
		
		self.depth = src_arrs.shape[self.axis]
		self.tgt_fns = tgt_fns
		
		del src_arrs

	def __len__(self):
		return len(self.src_arrs)*self.depth
		
	def __getitem__(self,idx:int):
		#Find image and slice
		idx_img = idx // self.depth
		idx_slice = idx % self.depth

		#Load image, parcellation, and task (optional)
		src_arr = nb.load(self.src_arrs[idx_img]).get_fdata(dtype=np.float32)
		tgt_arr = nb.load(self.tgt_fns[idx_img]).get_fdata(dtype=np.float32)
		task = self.tasks[idx_img]

		#Swap axes
		src_arr = src_arr.swapaxes(0,self.axis)[idx_slice]
		tgt_arr = tgt_arr.swapaxes(0,self.axis)[idx_slice]

		sample = (src_arr[None,:,:], tgt_arr[None,:,:], task, idx_slice)

		return sample
		


'''
...Usefull function to find data that meet the target criteria
'''
def find(data:list, target:list):

	bin_ = []
	for t in data:
		if t in target:
			bin_.append(True)
		else:
			bin_.append(False)
			
	return bin_