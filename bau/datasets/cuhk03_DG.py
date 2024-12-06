from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re

from ..utils.data import BaseImageDataset


class CUHK03_DG(BaseImageDataset):
    """
    CUHK03
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!
    Dataset statistics:
    # identities: 1360
    # images: 13164
    # cameras: 2
    """

    dataset_dir = 'cuhk03-np'

    def __init__(self, root, verbose=True, **kwargs):
        super(CUHK03_DG, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'detected/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'detected/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'detected/bounding_box_test')

        self._check_before_run()

        train = self._process_dir_DG({'train': self.train_dir,
                                      'query': self.query_dir,
                                      'gallery': self.gallery_dir}
                                     , relabel=True)
        self.train = train

        if verbose:
            print("=> CUHK03-NP(detected) loaded")
            self.print_dataset_statistics_DG(train)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 2
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_dir_DG(self, dir_paths, relabel=False):
        img_paths_total = []
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for name, dir_path in dir_paths.items():
            if name != 'train': name = 'test'
            img_paths = glob.glob(osp.join(dir_path, '*.png'))
            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                pid = name + str(pid)
                pid_container.add(pid)
            img_paths_total += img_paths
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for name, dir_path in dir_paths.items():
            if name != 'train': name = 'test'
            img_paths = glob.glob(osp.join(dir_path, '*.png'))
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                pid = name + str(pid)
                assert 1 <= camid <= 2
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, pid, camid))

        return dataset

