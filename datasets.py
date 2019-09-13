import torch
from torch.utils.data import Dataset
import json
import os
import io
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, path, data_type):
        """
        :param path: folder where data files are stored
        :param data_type: split, one of 'train' or 'test'
        """
        self.data_type = data_type

        assert self.data_type in {'train', 'test'}

        self.path = path

        # Read data files
        with open(os.path.join(path, self.data_type + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(path, self.data_type + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        with open(self.images[i], "rb") as f:
            b = io.BytesIO(f.read())
            image = Image.open(b)
            image = image.convert('RGB')

            # Read objects in this image (bounding boxes, labels, difficulties)
            objects = self.objects[i]
            boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
            labels = torch.LongTensor(objects['labels'])  # (n_objects)

            # Apply transformations
            image, boxes, labels = transform(image, boxes, labels, data_type=self.data_type)

        return image, boxes, labels

    def __len__(self):
        return len(self.images)

def collate(batch):
        """
        Describes how to combine tensors of different sizes when we have images with various number of objects

        :param batch: an iterable of N sets
        :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
        """

        imgs = []
        boxes = []
        labels = []

        for b in batch:
            imgs.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(imgs, dim=0)

        return images, boxes, labels  # 3 lists of N tensors each
