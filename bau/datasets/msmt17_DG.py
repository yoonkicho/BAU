import glob
import re
import os.path as osp
import urllib
import zipfile

from ..utils.osutils import mkdir_if_missing
from ..utils.data import BaseImageDataset


class MSMT17_DG(BaseImageDataset):
    dataset_dir = 'MSMT17'

    def __init__(self, root='./dataset', verbose=True, **kwargs):
        super(MSMT17_DG, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://www.pkuvmc.com/publications/msmt17.html'
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._download_data()
        self._check_before_run()

        train = self._process_dir_DG({'train': self.train_dir,
                                      'query': self.query_dir,
                                      'gallery': self.gallery_dir}
                                     , relabel=True)
        self.train = train

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics_DG(train)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading MSMT17 dataset")
        urllib.request.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

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
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 15
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_dir_DG(self, dir_paths, relabel=False):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        pid_container = set()

        for name, dir_path in dir_paths.items():
            if name != 'train': name = 'test'
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                pid = name + str(pid)
                pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for name, dir_path in dir_paths.items():
            if name != 'train': name = 'test'
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                pid = name + str(pid)
                assert 1 <= camid <= 15
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, pid, camid))

        return dataset
