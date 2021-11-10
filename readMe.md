# 576 Final Project

## Project Organization

```
root directory
	├── /train                 # unziped data from kaggle
	├── /test                  # unziped data from kaggle
	├── /train                 # unziped data from kaggle
	├── /LIVECell_dataset_2021 # unziped data from kaggle
	├── /train_reorganize      # run reorganize_images.py to generate
	----------------------------------------------------------------------------
	├── reorganize_images.py   # see Data Reorganization part
	├── utils.py               # useful helper function, check it out
	├── data_generator.py      # data generator for training model
	├── Unet.py                # Unet model
	├── EDA.ipynb              # Exploratory Data Analysis
	└── classifier.py          # classification model
```

### Data Collection
Please download [dataset](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data)
and unzip in the root directory.

### Data Reorganization
Run `python reorganize_images.py`. This file will copy file from `./train` into `./train_reorganize/cell_type` of the 
three cell type : cort, astro and shsy5y. Also decode the annotation mask into
one PNG file with the same file name as the original image file. Merge LIVECell's shsy5y images 
along with masks to the training data. This file is **safe** to run multiple times.

## Work Assignment

### Classifier

This part is assigned to **PERSON**

Please finish the API in `classifier.py`. We need a CNN based classifier, which tasks in
a test image, and output classification of this image. Please contain the link to trained
model weight.

### UNet Segmentation

This part is assigned to Tianxing Wang

Train a UNet model for segmentation


## Continue...