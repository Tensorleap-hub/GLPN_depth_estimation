import os
import yaml


# yaml_path = os.path.normpath(os.path.join(os.getcwd(), 'config.yml'))
root = os.path.abspath(os.path.dirname(__file__))
yaml_path = os.path.join(root, 'config.yaml')
# yaml_path = "../config.yaml"
yaml_config = yaml.safe_load(open(yaml_path))


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

gcs_config = Struct(**yaml_config["GCS"])
data_config = Struct(**yaml_config["Data"])
kitti_cat = yaml_config["KittiCat"]



