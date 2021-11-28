from utils import *
import os
from tqdm.notebook import tqdm
import shutil
from glob import glob

df_train = pd.read_csv('train.csv')
df_train.drop(columns=['elapsed_timedelta'], inplace=True)
print(f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')


def parse_filename(filename):
    image_id = filename.split('.')[0]
    cell_type = filename.split('[')[0]
    filename_split = filename.split('_')
    plate_time = filename_split[-3]
    sample_date = filename_split[-4]
    sample_id = '_'.join(filename_split[:3]) + '_' + '_'.join(filename_split[-2:]).split('.')[0]

    return image_id, cell_type, plate_time, sample_date, sample_id


train_semi_supervised_images = os.listdir('train_semi_supervised')
for filename in tqdm(train_semi_supervised_images):
    image_id, cell_type, plate_time, sample_date, sample_id = parse_filename(filename)
    sample = {
        'id': image_id,
        'annotation': np.nan,
        'width': 704,
        'height': 520,
        'cell_type': cell_type,
        'plate_time': plate_time,
        'sample_date': sample_date,
        'sample_id': sample_id
    }
    df_train = df_train.append(sample, ignore_index=True)

df_train['cell_type'] = df_train['cell_type'].str.rstrip('s')
print(
    f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

def decode_rle_mask(rle_mask, shape):

    """
    Decode run-length encoded segmentation mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded segmentation mask string
    shape (tuple): Height and width of the mask

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
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


# Final dataframe with unannotated images and extracted metadata is saved as a csv file.
for image_id in tqdm(df_train.loc[~df_train['annotation'].isnull(), 'id'].unique()):

    image = cv2.imread(f'train/{image_id}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    df_train.loc[df_train['id'] == image_id, 'image_mean'] = np.mean(image)
    df_train.loc[df_train['id'] == image_id, 'image_std'] = np.std(image)

    for rle_mask in df_train.loc[df_train['id'] == image_id, 'annotation']:
        mask = decode_rle_mask(rle_mask, (520, 704))
        df_train.loc[(df_train['id'] == image_id) & (df_train['annotation'] == rle_mask), 'mask_area'] = np.sum(mask)

for image_id in tqdm(df_train.loc[df_train['annotation'].isnull(), 'id'].unique()):
    image = cv2.imread(f'train_semi_supervised/{image_id}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    df_train.loc[df_train['id'] == image_id, 'image_mean'] = np.mean(image)
    df_train.loc[df_train['id'] == image_id, 'image_std'] = np.std(image)

annotation_counts = df_train.loc[~df_train['annotation'].isnull()].groupby('id')['annotation'].count()
df_train['annotation_count'] = df_train['id'].map(annotation_counts)
df_train.to_csv('train_processed.csv', index=False)




