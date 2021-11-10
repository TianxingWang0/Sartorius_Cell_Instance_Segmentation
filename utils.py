"""

Useful helper methods

"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO

WIDTH = 704
HEIGHT = 520
IMAGE_SHAPE = (HEIGHT, WIDTH)


def load_train_csv(file_name, verbose=True):
    """
    return loaded dataframe of train.csv file
    :param file_name:
    :param verbose:
    :return:
    """
    train_df = pd.read_csv(file_name)
    train_df.drop(columns=['elapsed_timedelta', 'width', 'height', 'plate_time', 'sample_date'], inplace=True)
    if verbose:
        print("{} train images - {} annotations".format(train_df['id'].nunique(), len(train_df['id'])))
        print("Columns : {}".format(train_df.columns.values))
    return train_df


def decode_rle_to_mask(rle_mask, shape):
    """
    return decoded run length mask into 2-D np.array as a graph
    :param rle_mask:
    :param shape:
    :return:
    """
    rle_mask = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    mask = mask.reshape(shape[0], shape[1])
    return mask


def encode_mask_to_rle(mask):
    """
    return encoded run length mask from 2-D np.array mask for submission
    :param mask:
    :return:
    """
    dots = np.where(mask.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


def merge_masks(df, image_id):
    masks = []
    for mask in df.loc[df['id'] == image_id, 'annotation'].values:
        decoded_mask = decode_rle_to_mask(rle_mask=mask, shape=IMAGE_SHAPE)
        masks.append(decoded_mask)
    annotation_count = len(masks)
    mask = np.stack(masks)
    mask = np.any(mask == 1, axis=0)
    return mask, annotation_count


def visualize_image_with_mask(image, mask):
    fig, axes = plt.subplots(figsize=(20, 20), ncols=2)
    fig.tight_layout(pad=5.0)

    axes[0].imshow(image, cmap='gray')
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, alpha=0.4)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    # axes[0].set_title(f'{image_path} - {cell_type} - {int(annotation_count)}', fontsize=20, pad=15)
    axes[1].set_title('Segmentation Mask', fontsize=20, pad=15)
    plt.show()
    plt.close(fig)


def visualize_image_id(df, image_id):
    """
    Visualize image along with segmentation masks

    Parameters
    ----------
    df [pandas.DataFrame of shape (73585, 9)]: Training dataframe
    image_id (str): Image ID (filename)
    """

    image_path = df.loc[df['id'] == image_id, 'id'].values[0]
    cell_type = df.loc[df['id'] == image_id, 'cell_type'].values[0]

    image = cv2.imread('./train/{}.png'.format(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask, annotation_count = merge_masks(df, image_id)
    visualize_image_with_mask(image, mask)


def train_history_plot(acc_list, loss_list, test_acc_list, title="", converge=0.0):
    """

    :param acc_list:
    :param loss_list:
    :param test_acc_list:
    :param title: The title for the plot
    :param converge: The converge validation accuracy, plot a horizontal line
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    epochs = len(loss_list);

    plt.xlim(1, epochs + 1)

    acc_curve, = ax.plot(range(1, epochs + 1), acc_list, '-k', linewidth=1, color='r')
    test_acc_curve, = ax.plot(range(1, epochs + 1), test_acc_list, '--k', linewidth=1, color='b')

    ax2 = ax.twinx()
    loss_curve, = ax2.plot(range(1, epochs + 1), loss_list, 'k', linewidth=1, color='g')

    if title:
        plt.title(title)
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    if converge > 0.0:
        ax.hlines(converge, 1, len(loss_list), colors="y", linestyles="dashed")

    plt.legend(handles=[acc_curve, test_acc_curve, loss_curve], labels=['train acc', 'test acc', 'loss'],
               loc='lower right')

    plt.show()


def load_json_to_dict(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def modify_live_cell_json(file_path):
    """
    the json file in LIVECELL need to be modified before decode with coco
    :param file_path:
    :return:
    """
    data = load_json_to_dict(file_path)
    annotations = data['annotations']
    if isinstance(annotations, dict):  # original is dict, which should be modified as list
        data['annotations'] = [annotations[ann] for ann in annotations]
        with open(file_path, 'w') as j: # overwrite original json data file
            json.dump(data, j)


def get_coco(file_path):
    """

    :param file_path:
    :return:
    """
    modify_live_cell_json(file_path)
    return COCO(file_path)


def decode_coco_annotation_to_mask(coco, image_id, image_size=IMAGE_SHAPE):
    """

    :param coco:
    :param image_id:
    :param image_size:
    :return:
    """
    annotation_ids = coco.getAnnIds(image_id)
    annotations = coco.loadAnns(annotation_ids)
    mask = np.zeros(image_size)
    for ann in annotations:
        mask = np.maximum(coco.annToMask(ann), mask)
    return mask
