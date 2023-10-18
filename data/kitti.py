from typing import List, Dict
import os
import pandas as pd

from GLPN.configs import gcs_config, data_config, kitti_cat
from utils.gcs_utils import _download


class Kitti:
    def __init__(self, left_camera: bool = True):
        self._bucket = gcs_config.bucket_name
        self._project_id = gcs_config.project_id
        self._path_dir = gcs_config.data_dir
        # self._image_folder_num = 'image_02' if left_camera else 'image_03'
        self._kitti_cats: dict = kitti_cat
        self.train_df = self._fpaths_to_df(os.path.join(gcs_config.data_dir, gcs_config.paths_dir, gcs_config.fname_train_paths))
        self.val_df = self._fpaths_to_df(os.path.join(gcs_config.data_dir, gcs_config.paths_dir, gcs_config.fname_test_paths))

    def _fpaths_to_df(self, fpath: str):
        local_fpath = _download(fpath)
        df = pd.read_csv(local_fpath, sep=' ', names=['image', 'gt'])
        df['folder'] = df['image'].apply(lambda x: x.split('/')[3])
        df = df[df['folder'].isin(self._kitti_cats.keys())]
        df['category'] = df['folder'].replace(self._kitti_cats)
        df['image'] = df['image'].apply(lambda x: '/'.join(['/depth_estimation']+x.split('/')[3:]))
        df['gt'] = df['gt'].apply(lambda x: '/'.join(['']+x.split('/')[2:]))
        return df

    # def get_raw_data_path(self, sub: str, drive_folder: str, ind: int):
    #     return os.path.join(self._path_dir,
    #                         sub,
    #                         drive_folder,
    #                         self._image_folder_num,
    #                         'data',
    #                         (str(ind).zfill(10) + ".png"))
    #
    # def get_gt_path(self, sub: str, drive_folder: str, ind: int):
    #     return os.path.join(self._path_dir,
    #                         sub,
    #                         drive_folder,
    #                         'proj_depth',
    #                         'groundtruth',
    #                         self._image_folder_num,
    #                         (str(ind).zfill(10) + ".png"))

    # def _build_IDs(self):
    #     """Build the list of samples and their IDs, split them in the proper datasets.
    #     Called by the base class on init.
    #     Each ID is a tuple.
    #     For the training/val datasets, they look like ('000065_10.png', '000065_11.png', '000065_10.png')
    #      -> gt flows are stored as 48-bit PNGs
    #     For the test dataset, they look like ('000000_10.png', '00000_11.png', '000000_10.flo')
    #     """
    #     # Search the train folder for the samples, create string IDs for them
    #
    #     bucket = _connect_to_gcs_and_return_bucket(self._bucket)
    #     input_frames, gt_frames = [], []
    #     self.train_IDs, self.val_IDs = [], []
    #     self.train_cats, self.val_cats = [], []
    #     for sub, dir in zip(['train', 'val'], [gcs_config.train_folders, gcs_config.val_folders]):
    #         for folder in dir:
    #             gt_frames = [obj.name for i, obj in enumerate(bucket.list_blobs(prefix=os.path.dirname(self.get_gt_path(sub, folder, 0)))) if i < data_config.n_frames]
    #             input_frames = [os.path.join(os.path.dirname(self.get_raw_data_path(sub, folder, 0)), gt_frames[i].split('/')[-1]) for i in range(len(gt_frames))]
    #             if sub == 'train':
    #                 self.train_IDs += [(input_frames[i], gt_frames[i]) for i in range(0, len(input_frames))]
    #                 self.train_cats += [self._kitti_cats[folder]]*len(input_frames)
    #             else:
    #                 self.val_IDs += [(input_frames[i], gt_frames[i]) for i in range(0, len(input_frames))]
    #                 self.val_cats += [self._kitti_cats[folder]]*len(input_frames)


    def get_kitti_data(self) -> Dict[str, List[str]]:
        data_dict = {'train': {}, "validation": {}}
        TRAIN_SIZE = data_config.train_size
        VAL_SIZE = data_config.val_size
        data_dict['train'] = self.train_df.sample(min(TRAIN_SIZE, len(self.train_df)), ignore_index=True)
        data_dict['validation'] = self.val_df.sample(min(VAL_SIZE, len(self.val_df)), ignore_index=True)
        return data_dict


