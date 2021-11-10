"""

In the Sartorious dataset, one image only contains one type of cells, reorganize the original data directory
by classify images in to 3 folders as their own cell type kind.

Merge annotations of one image and store as a mask image

There are extra data of cell type shsy5y in dataset LIVECell, this script
also select these images into /train_reorganize

"""

import os
import shutil
from utils import *

train_df = load_train_csv('train.csv')

if not os.path.exists("train_reorganize"):
    os.mkdir("train_reorganize")

print("Copy image files to different folders")
# copy image files to different folders
for cell_type in train_df.cell_type.value_counts().keys():
    target_root_path = os.path.join("train_reorganize", cell_type)
    if not os.path.exists(target_root_path):
        os.mkdir(target_root_path)
    image_ids = train_df[train_df.cell_type == cell_type]['id'].drop_duplicates().values
    for image_id in image_ids:
        image_id += '.png'
        target_path = os.path.join(target_root_path, image_id)
        if not os.path.exists(target_path):  # in case of multiple copy
            shutil.copyfile(os.path.join("train", image_id), target_path)

print("Merge annotations into one image and save")
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
        target_path = os.path.join(target_root_path, image_id + '.png')
        if not os.path.exists(target_path):  # in case of multiple copy
            mask, annotation_count = merge_masks(train_df, image_id)
            cv2.imwrite(target_path, mask.astype(float))

print("Merge LIVECell shsy5y cell type to the train_reorganize, may take a few minutes.")
# Merge LIVECell shsy5y cell type to the train_reorganize
root_src_path = os.path.join('LIVECell_dataset_2021', 'images', 'livecell_train_val_images', 'SHSY5Y')
coco_train = get_coco('./LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_train.json')
coco_val = get_coco('./LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_val.json')
# test dataset is labeled in LIVECell
coco_test = get_coco('./LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_test.json')

for coco in [coco_train, coco_val, coco_test]:
    if coco is coco_test:
        root_src_path = os.path.join('LIVECell_dataset_2021', 'images', 'livecell_test_images', 'SHSY5Y')
    for img_id in coco.getImgIds():
        img_file_name = coco.imgs[img_id]['file_name']
        img_target_path = os.path.join('train_reorganize', 'shsy5y', img_file_name[:-3] + 'png') # convert from .itf to .png
        if not os.path.exists(img_target_path):
            img = cv2.cvtColor(cv2.imread(os.path.join(root_src_path, img_file_name)), cv2.COLOR_BGR2GRAY)
            cv2.imwrite(img_target_path, img)
        mask_task_path = os.path.join('train_reorganize', 'annotation', 'shsy5y', img_file_name[:-3] + 'png')
        if not os.path.exists(mask_task_path):
            mask = decode_coco_annotation_to_mask(coco, img_id)
            cv2.imwrite(mask_task_path, mask)


