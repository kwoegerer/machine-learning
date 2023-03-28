
import os
import glob
import numpy as np
from PIL import Image


class ImageNormalizer:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.absolute_input_dir = os.path.abspath(input_dir)
        self.file_paths = []
        self.n_file_paths = 0
        self.mean = None
        self.std = None

        for image_file in glob.glob(os.path.join(os.path.abspath(input_dir), "**/*"), recursive=True):
            filename, file_extension = os.path.splitext(image_file)
            if file_extension == '.jpg':
                self.file_paths.append(os.path.relpath(image_file, input_dir))
        self.file_paths = sorted(self.file_paths)
        self.n_file_paths = len(self.file_paths)

    def analyze_images(self):
        img_mean = np.zeros(self.n_file_paths)
        img_std = np.zeros(self.n_file_paths)
        i = 0
        for file_path in self.file_paths:
            image = Image.open(os.path.join(self.absolute_input_dir, file_path))
            image_data = np.asarray(image)
            image_data = np.float64(image_data)
            img_mean[i] = np.mean(image_data)
            img_std[i] = np.std(image_data)
            i += 1
        self.mean = img_mean.mean()
        self.std = img_std.mean()
        return img_mean.mean(), img_std.mean()

    def get_images_data(self):
        if self.mean is None or self.std is None:
            raise ValueError
        for file_path in self.file_paths:
            image = Image.open(os.path.join(self.absolute_input_dir, file_path))
            image_data = np.asarray(image)
            image_data = np.float32(image_data)
            normalized_image_data = (image_data - self.mean) / self.std
            yield normalized_image_data
