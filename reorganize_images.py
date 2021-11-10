"""

In the Sartorious dataset, one image only contains one type of cells, reorganize the original data directory
by classify images in to 3 folders as their own cell type kind.

Merge annotations of one image and store as a mask image

"""

import numpy as np
import pandas as pd
import os
import shutil
from utils import *

train_df = load_train_csv('train.csv')

if not os.path.exists("train_reorganize"):
    os.mkdir("train_reorganize")

# copy image files to different folders
for cell_type in train_df.cell_type.value_counts().keys():
    target_root_path = os.path.join("train_reorganize", cell_type)
    if not os.path.exists(target_root_path):
        os.mkdir(target_root_path)
    image_ids = train_df[train_df.cell_type == cell_type]['id'].drop_duplicates().values
    for image_id in image_ids:
        image_id += '.png'
        file_path = os.path.join("train", image_id)
        target_path = os.path.join(target_root_path, image_id)
        if not os.path.exists(target_path):  # in case of multiple copy
            shutil.copyfile(file_path, target_path)

# merge annotations into one image and save
for cell_type in train_df.cell_type.value_counts().keys():
    target_root_path = os.path.join("train_reorganize", "annotation")
    if not os.path.exists(target_root_path):
        os.mkdir(target_root_path)
    target_root_path = os.path.join(target_root_path, cell_type)
    if not os.path.exists(target_root_path):
        os.mkdir(target_root_path)
    image_ids = train_df[train_df.cell_type == cell_type]['id'].drop_duplicates().values
    for image_id in image_ids:
        mask, annotation_count = merge_masks(train_df, image_id)
        image_id += '.png'
        target_path = os.path.join(target_root_path, image_id)
        if not os.path.exists(target_path):  # in case of multiple copy
            cv2.imwrite(target_path, mask.astype(float))
