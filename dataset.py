import imageio as iio
import numpy as np
from tqdm import tqdm
import os

class Dataset:
    
    path_to_dataset = "./flower/"
    files_list = []
    dataset = []
    image_shape = [0, 0]
    total_images = 0
    
    def __init__(self, path = None):
        if path != None:
            self.path_to_dataset = path
        self.file_list()
    
    def file_list(self):
        self.files_list = os.listdir(self.path_to_dataset)
        self.total_images = len(self.files_list)
        return self.files_list
    
    def read_image(self, image_name):
        return iio.imread(self.path_to_dataset + image_name)
    
    def load_data(self):
        for file in self.files_list:
            self.dataset.append([self.read_image(file)])
        self.dataset = np.array(self.dataset)
        self.image_shape = np.array(file).shape
        
        self.dataset = np.squeeze(self.dataset) / 255
        
        return self.dataset