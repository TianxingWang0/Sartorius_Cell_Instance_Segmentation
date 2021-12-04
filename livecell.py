# Get the cell segmentation mask of LIVECell


import os
import shutil
from utils import *

if not os.path.exists('livecell_reorganize'):
    os.mkdir('livecell_reorganize')

if not os.path.exists(os.path.join("livecell_reorganize", "image")):
    os.mkdir(os.path.join("livecell_reorganize", "image"))

if not os.path.exists(os.path.join("livecell_reorganize", "annotation")):
    os.mkdir(os.path.join("livecell_reorganize", "annotation"))

cell_types = [cell_type for cell_type in os.listdir(os.path.join('LIVECell_dataset_2021', 'annotations', 'LIVECell_single_cells'))]

for cell_type in cell_types:
    if cell_type != 'shsy5y':
        print("Reorganizing LIVECell {} cell type.".format(cell_type))
        if not os.path.exists(os.path.join("livecell_reorganize", "annotation", cell_type)):
            os.mkdir(os.path.join("livecell_reorganize", "annotation", cell_type))
        if not os.path.exists(os.path.join("livecell_reorganize", "image", cell_type)):
            os.mkdir(os.path.join("livecell_reorganize", "image", cell_type))
        root_src_path = os.path.join('LIVECell_dataset_2021', 'images', 'livecell_train_val_images', cell_type.upper())
        coco_train = get_coco('./LIVECell_dataset_2021/annotations/LIVECell_single_cells/{}/livecell_{}_train.json'.format(cell_type, cell_type))
        coco_val = get_coco('./LIVECell_dataset_2021/annotations/LIVECell_single_cells/{}/livecell_{}_val.json'.format(cell_type, cell_type))
        # test dataset is labeled in LIVECell
        coco_test = get_coco('./LIVECell_dataset_2021/annotations/LIVECell_single_cells/{}/livecell_{}_test.json'.format(cell_type, cell_type))

        for coco in [coco_train, coco_val, coco_test]:
            if coco is coco_test:
                root_src_path = os.path.join('LIVECell_dataset_2021', 'images', 'livecell_test_images', cell_type.upper())
            for img_id in coco.getImgIds():
                img_file_name = coco.imgs[img_id]['file_name']
                img_target_path = os.path.join('livecell_reorganize', 'image', cell_type,
                                               img_file_name[:-3] + 'png')  # convert from .itf to .png
                if not os.path.exists(img_target_path):
                    img = cv2.cvtColor(cv2.imread(os.path.join(root_src_path, img_file_name)), cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(img_target_path, img)
                mask_task_path = os.path.join('livecell_reorganize', 'annotation', cell_type, img_file_name[:-3] + 'png')
                if not os.path.exists(mask_task_path):
                    mask = decode_coco_annotation_to_mask(coco, img_id)
                    cv2.imwrite(mask_task_path, mask)
