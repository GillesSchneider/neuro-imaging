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
		
		#Create a folder to save parcellations in the current folder
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
		idx_img = idx // self.depth
		idx_slice = idx % self.depth

		src_arr = nb.load(self.src_arrs[idx_img]).get_fdata(dtype=np.float32)
		tgt_arr = nb.load(self.tgt_fns[idx_img]).get_fdata(dtype=np.float32)
		task = self.tasks[idx_img]

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