import os
import random
from glob import glob

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def yolo2xyxy(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2]/2, bboxes[1] - bboxes[3]/2
    xmax, ymax = bboxes[0] + bboxes[2]/2, bboxes[1] + bboxes[3]/2
    return xmin, ymin, xmax, ymax

def read_yolo_annotation(filename):
    with open(filename,'r') as f:
        bboxes = []
        labels = []
        label_lines = f.readlines()
        for label_line in label_lines:
            label, x_c, y_c, w, h = label_line.split(' ')
            x_c = float(x_c)
            y_c = float(y_c)
            w = float(w)
            h = float(h)
            label = int(label)

            yolo_bbox = [x_c, y_c, w, h]

            bboxes.append(yolo_bbox)
            labels.append(label)
    return bboxes, labels

def detection_collate_fn(raw_batch):

    images = []      # [B, H, W, C] or [B, C, H, W]
    annotations = [] # [B, ND*6], ND: #detections

    for i, _ in enumerate(raw_batch):
        image = raw_batch[i][0]
        annotation =  raw_batch[i][1]

        annotation = np.array(annotation).reshape(-1,6)

        images.append(image)
        annotations.append( np.array(annotation).reshape(-1,6) )

    batch_images = torch.from_numpy(np.array( images ))
    batch_annotations = torch.from_numpy( np.vstack( annotations )  )

    return batch_images, batch_annotations


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, transforms=None, channels_first=True, **kwargs):
        super(YOLODataset).__init__(**kwargs)

        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.all_images_path = sorted(glob( os.path.join(images_dir, '*.jpg') ))
        self.all_labels_path = sorted(glob( os.path.join(labels_dir, '*.txt') ))

        self.channels_first = channels_first
        self.class_names = class_names
        self.transforms = transforms
        self.colors = np.random.uniform( 0, 255, size=(len(class_names), 3) )

    def __len__(self):
        return len(self.all_images_path)

    def __getitem__(self, index, use_transformations=True):
        image_path = self.all_images_path[index]
        label_path = self.all_labels_path[index]

        # Read from disk as numpy on RGB format
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # annotations
        bboxes, labels =  read_yolo_annotation(label_path)

        bboxes = np.array(bboxes).reshape(-1, 4)
        labels = np.array(labels).reshape(-1, 1)
        indexes= np.array([index]*len(bboxes)).reshape(-1, 1)

        if (use_transformations == False) or (self.transforms == None):
            annotations = np.concatenate( [indexes, labels, bboxes], axis=-1)
            if self.channels_first:
                image = np.transpose(image, axes=[2, 0, 1])
            return tuple([image, annotations])

        # Augmentation from albumentation lib
        transformed = self.transforms(image=image, bboxes=bboxes, labels=labels, indexes=indexes)

        image  = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['labels']
        indexes= transformed['indexes']

        annotations = np.concatenate( [indexes, labels, bboxes], axis=-1)

        if self.channels_first:
            image = np.transpose(image, axes=[2, 0, 1])
        return tuple([image, annotations])

    def plot_box(self, image, bboxes, labels):
        # Need the image height and width to denormalize
        # the bounding box coordinates
        height, width = image.shape[:2]
        # lw = max(round(sum(image.shape / 2 * 0.003)), 2) # line width
        lw = max(round( sum(image.shape) / 2 * 0.003 ),2)
        tf = max(lw - 1, 1)
        for box_num, box in enumerate(bboxes):
            x1, y1, x2, y2 = yolo2xyxy(box)
            # denormalize the coordinates
            xmin = int(x1*width)
            ymin = int(y1*height)
            xmax = int(x2*width)
            ymax = int(y2*height)

            p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

            class_name = self.class_names[int(labels[box_num])]

            color = self.colors[self.class_names.index(class_name)]

            cv2.rectangle(image, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)

            # for filled rectangle
            w, h = cv2.getTextSize(class_name,
                                0,
                                fontScale=lw/3,
                                thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(image, p1, p2, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(image,
                        class_name,
                        (p1[0], p1[1] -5 if outside else p1[1] + h + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=lw/3.5,
                        color=(255,255,255),
                        thickness=tf,
                        lineType=cv2.LINE_AA
                    )
        return image

    def plot(self, num_samples=25, plot_transformed_data=True):

        # temp = list( zip(self.all_images_path, self.all_labels_path) )
        # random.shuffle(temp)
        # all_images_path, all_labels_path = zip(*temp)
        # all_images_path, all_labels_path = list( all_images_path ), list( all_labels_path )
        random_indexes = np.random.randint(0, len(self.all_images_path), size=num_samples)

        nrows = int(np.sqrt(num_samples))

        plt.figure(figsize=(15,12))
        for i, rand_img_idx in enumerate(random_indexes):
            # Read from disk as numpy on RGB format
            if plot_transformed_data:
                image, annotations = self.__getitem__(rand_img_idx, plot_transformed_data)
                image = np.transpose(image, axes=[1, 2, 0])
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            else:
                image = cv2.cvtColor(cv2.imread(self.all_images_path[rand_img_idx]), cv2.COLOR_BGR2RGB)
                bboxes, labels = read_yolo_annotation(self.all_labels_path[rand_img_idx])

            labels = annotations[:,1]
            bboxes = annotations[:,2:]

            result_image = self.plot_box(image, bboxes, labels)
            plt.subplot(nrows, nrows, i+1) # visualize 2x2 grid of images
            plt.imshow(result_image)
            plt.axis('off')
        plt.tight_layout()
        plt.show()