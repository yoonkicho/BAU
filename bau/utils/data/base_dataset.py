# encoding: utf-8
import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for info in data:           
            pids += [info[1]]
            cams += [info[2]]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
    
    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def print_dataset_statistics_DG(self, train):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  ----------------------------------------")

    
class MultiSourceTrainDataset(BaseImageDataset):
    def __init__(self, datasets, verbose=True, **kwargs):
        super(MultiSourceTrainDataset, self).__init__()
        self.datasets = datasets
        self._check_before_run()

        train = self._merge_data(datasets)

        if verbose:
            print("=> Multi-Source Dataset loaded")
            self.print_dataset_statistics_DG(train)

        self.train = train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not len(self.datasets) > 0:
            raise RuntimeError("More than one source training dataset is required.")

    def _merge_data(self, datasets):
        merged_dataset = []
        pid_offset = 0
        cid_offset = 0
        did_offset = 0
        for dataset in datasets:
            train_dataset = sorted(dataset.train)
            for i, (img_path, pid, cid) in enumerate(train_dataset):
                merged_dataset.append((img_path, pid + pid_offset, cid + cid_offset, did_offset))
            pid_offset += dataset.num_train_pids
            cid_offset += dataset.num_train_cams
            did_offset += 1

        return merged_dataset
