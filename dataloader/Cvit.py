import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from . import preprocess 
import random
from itertools import chain
import time
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
	tries = 2
	for i in range(tries):
		try:
			img = Image.open(path).convert('RGB')
		except OSError as e:
			if i < tries - 1: # i is zero indexed
				continue
			else:
				print(path)
				return None
	return img


class myImageloader(data.Dataset):
	def __init__(self, datapath='./custom_data', loader=default_loader, video_list = None):
		# video list makes the data list for only those videos to be included
		# if it isn't specified, default is for all

		frames_dict = {}
		# print("in obj")
		frames_path = os.path.join(datapath, 'frames')
		dataset_names = os.listdir(frames_path)

		for dataset in dataset_names:
			dataset_path = os.path.join(frames_path, dataset)
			frames_dict[dataset] = os.listdir(dataset_path)

		# print(frames_dict)
		self.frames_dict = frames_dict
		self.frames_path = frames_path
		#Creating a list for get_item
		self.all_data_list = []
		#this dictionary keeps count of indexes from each frame
		#each entry is of the form:
		# [l,r) (not inclusive of r)
		# this is useful for list slicing when we do:
		# list[l:r]

		self.dataset_count = {}

		if video_list is None:
			count = 0
			for i in frames_dict.keys():
				self.dataset_count[i] = [count]
				for j in frames_dict[i]:
					self.all_data_list.append((i, j))
					count += 1
				self.dataset_count[i].append(count)

		else:
			count = 0
			for vid in video_list:
				self.dataset_count[vid] = [count]
				for j in frames_dict[vid]:
					self.all_data_list.append((vid, j))
					count += 1
				self.dataset_count[vid].append(count)

		# print(self.all_data_list)
		# print(self.dataset_count)

		# self.alldatalist = []
		# fp = open (datapath,"r")
		# for line in fp.readlines():
		#     line = line.strip()
		#     #print(line)
		#     self.alldatalist.append(line)
		# fp.close()
		# #print(len(self.alldatalist))

		self.loader = loader



	def __getitem__(self, index):
		(vid, img) = self.all_data_list[index]
		idx_l, idx_r = self.dataset_count[vid]
		idx_img_2 = np.random.randint(idx_l, idx_r-1)
		while idx_img_2 == index:
			idx_img_2 = np.random.randint(idx_l, idx_r-1)

		img1_path = os.path.join(self.frames_path, vid,img)
		# print(img1_path, os.path.isfile(img1_path))
		img1 = self.loader(img1_path)
		processed1 = preprocess.get_transform(augment=True)
		img1 = processed1(img1)

		if np.random.rand() > 0.5:
			img2 = transforms.functional.hflip(img1)
		else:
			# print("in second one")
			(vid_2, img_2) = self.all_data_list[idx_img_2]
			img2_path = os.path.join(self.frames_path, vid_2, img_2)
			img2 = self.loader(img2_path)
			# print(img2_path, os.path.isfile(img2_path))
			processed2 = preprocess.get_transform(augment=True)
			img2 = processed2(img2)

		return img1, img2 




		# identity_dir = self.alldatalist[index] + '/'
		# split_path = identity_dir.split('/')       
		# id_img_list = [identity_dir+'/'+img.name for img in os.scandir(identity_dir) if is_image_file(img.name)]

		# #random select one image
		# img_idx = np.random.randint(0, len(id_img_list)-1)
		# img_idx2 = np.random.randint(0, len(id_img_list)-1)        

		# img1 = self.loader(id_img_list[img_idx])

		# while img1 is None:
		# 	img_idx = np.random.randint(0, len(id_img_list)-1)
		# 	img1 = self.loader(id_img_list[img_idx])

		# processed1 = preprocess.get_transform(augment=True)  
		# img1 = processed1(img1)
					  
		# if np.random.rand() > 0.5:
		# 	img2 = transforms.functional.hflip(img1) 
		# else:
		# 	img2 = self.loader(id_img_list[img_idx2])
		# 	while img2 is None:
		# 		img_idx2 = np.random.randint(0, len(id_img_list)-1)      
		# 		img2 = self.loader(id_img_list[img_idx2])  
		# 	processed2 = preprocess.get_transform(augment=True)              
		# 	img2 = processed2(img2)
			 
		return img1, img2


	def __len__(self):
		return len(self.all_data_list)