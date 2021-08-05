from pascal_voc_writer import Writer
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob
import time
from shutil import move, copy
import random

# Arguments
itri_dir = '/home/julius/Datasets/ITRI_dataset/Dataset416/'
save_path = '/home/julius/Datasets/ITRI_dataset/'

# Valid classes dictionary (to rename if needed)
classes = {'0': 'pedestrian', '1': 'rider', '2': 'two-wheels', '3': 'unknown', '4': 'four-wheels'}
classes_keys = list(classes.keys())

# Number of test data
test_n = 3038

# Function to make folder
def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# # Convert polygon to bounding box
# # https://stackoverflow.com/questions/46335488/how-to-efficiently-find-the-bounding-box-of-a-collection-of-points
# def polygon_to_bbox(polygon):
#     x_coordinates, y_coordinates = zip(*polygon)
#     return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


# # Read a json file and convert to VOC format
# def read_json(file):
#     # if no relevant objects found in the image,
#     # don't save the xml for the image
#     relevant_file = False

#     data = []
#     with open(file, 'r') as f:
#         file_data = json.load(f)

#         for object in file_data['objects']:
#             label, polygon = object['label'], object['polygon']

#             # process only if label found in voc
#             if label in classes_keys:
#                 polygon = np.array([x for x in polygon])
#                 bbox = polygon_to_bbox(polygon)
#                 data.append([classes[label]] + bbox)

#         # if relevant objects found in image, set the flag to True
#         if data:
#             relevant_file = True

#     return data, relevant_file


def read_txt(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
    if len(lines) <= 0: return None, False
    num_of_objects = int(lines[0][:-1])
    for x in range(num_of_objects):
        data.append(lines[x+1][:-1].split())
    return data, True
        

# Function to save xml file
def save_xml(img_path, img_shape, data, save_path):
    writer = Writer(img_path, img_shape[0], img_shape[1])
    for element in data:
        writer.addObject(element[0], element[1], element[2], element[3], element[4])
    writer.save(save_path)


# Reading json files from each subdirectory
valid_files = []
train_files = []
test_files = []

# Make Annotations target directory if already doesn't exist
ann_dir = os.path.join(save_path, 'VOC2007', 'Annotations')
make_dir(ann_dir)

start = time.time()

# for category in os.listdir(cityscapes_dir_gt):
#     # no GT for test data
#     if category == 'test': continue

#     for city in os.listdir(os.path.join(cityscapes_dir_gt, category)):

#         # read files
#         files = glob.glob(os.path.join(cityscapes_dir, 'gtFine', category, city) + '/*.json')

#         # process json files
#         for file in files:
#             data, relevant_file = read_json(file)

#             if relevant_file:
#                 base_filename = os.path.basename(file)[:-21]
#                 xml_filepath = os.path.join(ann_dir, base_filename + '_leftImg8bit.xml')
#                 img_name = base_filename + '_leftImg8bit.png'
#                 img_path = os.path.join(cityscapes_dir, 'leftImg8bit', category, city,
#                                         base_filename + '_leftImg8bit.png')
#                 img_shape = plt.imread(img_path).shape
#                 valid_files.append([img_path, img_name])

#                 # make list of trainval and test files for voc format
#                 # lists will be stored in txt files
#                 trainval_files.append(img_name[:-4]) if category == 'train' else test_files.append(img_name[:-4])

#                 # save xml file
#                 save_xml(img_path, img_shape, data, xml_filepath)

n = 0
files = glob.glob(os.path.join(itri_dir, 'TXT') + '/*.txt')
for file in files:
    data, relevant_file = read_txt(file)
    if relevant_file:
        base_filename = os.path.basename(file)[:-4]
        xml_filepath = os.path.join(ann_dir, base_filename + '.xml')
        img_name = base_filename + '.jpg'
        img_path = os.path.join(itri_dir, 'RGB', img_name)
        img_shape = plt.imread(img_path).shape
        valid_files.append([img_path, img_name])
        # Make list of train and test files for VOC format
        test_files.append(img_name[:-4]) if n < test_n else train_files.append(img_name[:-4])
        # Save XML file
        save_xml(img_path, img_shape, data, xml_filepath)
        n = n+1


end = time.time() - start
print('Total Time taken (converting): ', end)

# Copy files into target path
images_savepath = os.path.join(save_path, 'VOC2007', 'JPEGImages')
make_dir(images_savepath)

start = time.time()
for file in valid_files:
    copy(file[0], os.path.join(images_savepath, file[1]))

end = time.time() - start
print('Total Time taken (copying): ', end)

# Create text files of trainval and test files
textfiles_savepath = os.path.join(save_path, 'VOC2007', 'ImageSets', 'Main')
make_dir(textfiles_savepath)

train_files_wr = [x + '\n' for x in train_files]
test_files_wr = [x + '\n' for x in test_files]

with open(os.path.join(textfiles_savepath, 'train.txt'), 'w') as f:
    f.writelines(train_files_wr)

with open(os.path.join(textfiles_savepath, 'test.txt'), 'w') as f:
    f.writelines(test_files_wr)
