import os
import shutil
from collections import defaultdict
from pathlib import Path

split_map = None
def get_split_map(split_txt_path):
    global split_map

    if split_map is None:
        split_map = defaultdict(list)

        with open(split_txt_path) as file:
            split_data = file.readlines()

        for line in split_data:
            split_line = line.split()
            split = int(split_line[1])
            if split == 0:
                split = "train"
            elif split == 1:
                split = "val"
            elif split == 2:
                split = "test"
            image_name = split_line[0]
            split_map[split].append(image_name)
    return split_map

split_labels_map = None
def get_split_labels_map(label_txt_path, split_txt_path):
    global split_labels_map
    if split_labels_map is None:
        split_labels_map = {}
        for split in ['train', 'val', 'test']:
            split_labels_map[split] = defaultdict(list)

        split_map = get_split_map(split_txt_path)
        assert Path(label_txt_path).exists(), "label_txt_path doesn't exist!!!"
        with open(label_txt_path) as file:
            label_data = file.readlines()

        for line in label_data:
            split_line = line.split()
            label = int(split_line[1])
            image_name = split_line[0]
            image_split = "train"

            for split, image_list in list(split_map.items())[1:]:
                if image_name in image_list:
                    image_split = split

            split_labels_map[image_split][label].append(image_name)

    return split_labels_map


def celeba_split_from_file(data_path, split_txt_path):
    """
    split dataset into train, val, test and places the images in corresponding folders based on split txt file.
    The file has the following format:
    `image.name.ext 0
     image.name.ext 1
     image.name.ext 2`

    Where 0, 1, and 2 are for train, val, and test.
    :param data_path: path where the images are stored
    :param split_txt_path: path where the the split_txt_file is stored
    :return: dict of lists containing image file names corresponding to each partition.
    keys are "train", "val", and "test"
    """
    print('\nSplitting images to train, val, and test partitions...')
    split_map = get_split_map(split_txt_path)
    for split, image_name in list(split_map.items()):
        src_path = os.path.join(data_path, image_name)

        if not os.path.exists(src_path):
            print("File not found", src_path)
            continue

        dst_path = os.path.join(data_path, split)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        img_dst_path = os.path.join(dst_path, image_name)

        shutil.copyfile(src_path, img_dst_path)
        os.remove(src_path)
    print("Splitting completed.\n")
    return split_map


def celeba_save_images_in_folders(data_path, label_txt_path, map=None):
    """
    saves images in config.DATASET_FOLDER_IMG in subfolders with names as their labels
    :param config:
    :return:
    """
    print('\nSaving images in folders...')

    with open(label_txt_path) as file:
        data = file.readlines()

    for line in data:
        split_line = line.split()
        label = int(split_line[1])
        image_name = split_line[0]
        if map is not None:
            partition = "train"
            if image_name in map['test']:
                partition = 'test'
            elif image_name in map['val']:
                partition = 'val'
            src_url = os.path.join(data_path, partition, image_name)
            dst_url_dir = os.path.join(data_path, partition, str(label))

        else:
            src_url = os.path.join(data_path, image_name)
            dst_url_dir = os.path.join(data_path, str(label))

        if not os.path.exists(src_url):
            print("Image", src_url, "not found")
            continue

        if not os.path.exists(dst_url_dir):
            os.mkdir(dst_url_dir)

        dst_url = os.path.join(dst_url_dir, image_name)

        shutil.copyfile(src_url, dst_url)
        os.remove(src_url)
    print('Images to folders completed\n')