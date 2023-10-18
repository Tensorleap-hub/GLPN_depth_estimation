from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
import numpy as np
# import cv2
from utils.gcs_utils import _download
from tl_helpers.preprocess import subset_images
from GLPN.GLPN import IMG_PROCESSOR
from GLPN.configs import data_config
# from tl_helpers.visualizers.visualizers import image_visualizer, loss_visualizer, mask_visualizer, cityscape_segmentation_visualizer
from tl_helpers.tl_utils import *

# from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
# from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
# from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt



# ----------------------------------- Input ------------------------------------------

def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['paths'][idx][0]#[idx % data["real_size"]]
    fpath = _download(str(cloud_path))
    img = Image.open(fpath).convert('RGB')
    # img = img.resize(data_config.image_size)
    img = IMG_PROCESSOR(img, return_tensors="pt")

    # img = np.array(Image.open(fpath).convert('RGB').resize(IMAGE_SIZE)) / 255.
    return img


# ----------------------------------- GT ------------------------------------------
from PIL import Image
import numpy as np




def gt_depth(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['paths'][idx][1]  # [idx % data["real_size"]]
    fpath = _download(str(cloud_path))
    depth_png = Image.open(fpath).resize(data_config.image_size)
    depth_png = np.array(depth_png, dtype=int)

    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth[..., np.newaxis]


# ----------------------------------- Metadata ------------------------------------------

def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    """ add TL index """
    return idx


def metadata_background_percent(idx: int, data: PreprocessResponse) -> float:
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(19.)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_percent_function_generator(class_idx: int):
    def get_metadata_percent(idx: int, data: PreprocessResponse) -> float:
        mask = get_categorical_mask(idx % data.data["real_size"], data)
        unique, counts = np.unique(mask, return_counts=True)
        unique_per_obj = dict(zip(unique, counts))
        count_obj = unique_per_obj.get(float(class_idx))
        if count_obj is not None:
            percent_obj = count_obj / mask.size
        else:
            percent_obj = 0.0
        return percent_obj

    get_metadata_percent.__name__ = Cityscapes.train_id_to_label[class_idx] + "_" + "class_percent"
    return get_metadata_percent


def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
    img = non_normalized_input_image(idx % data.data["real_size"], data)
    return np.mean(img)


def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx][0]


def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]


def metadata_dataset(idx: int, data: PreprocessResponse) -> str:
    return data.data['dataset'][idx % data.data["real_size"]]


def metadata_gps_heading(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsHeading']
    else:
        return DEFAULT_GPS_HEADING


def metadata_gps_latitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsLatitude']
    else:
        return DEFAULT_GPS_LATITUDE


def metadata_gps_longtitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsLongitude']
    else:
        return DEFAULT_GPS_LONGTITUDE


def metadata_outside_temperature(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['outsideTemperature']
    else:
        return DEFAULT_TEMP


def metadata_speed(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['speed']
    else:
        return DEFAULT_SPEED


def metadata_yaw_rate(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['yawRate']
    else:
        return DEFAULT_YAW_RATE


# ----------------------------------- Binding ------------------------------------------


leap_binder.set_preprocess(subset_images)

leap_binder.set_input(input_image, 'normalized_image')
leap_binder.set_ground_truth(gt_depth, 'mask')

leap_binder.set_metadata(metadata_filename, 'filename')
leap_binder.set_metadata(metadata_dataset, 'dataset')
leap_binder.set_metadata(metadata_idx, 'idx')

# leap_binder.set_metadata(metadata_city, DatasetMetadataType.string, 'city')
# leap_binder.set_metadata(metadata_gps_heading, DatasetMetadataType.float, 'gps_heading')
# leap_binder.set_metadata(metadata_gps_latitude, DatasetMetadataType.float, 'gps_latitude')
# leap_binder.set_metadata(metadata_gps_longtitude, DatasetMetadataType.float, 'gps_longtitude')
# leap_binder.set_metadata(metadata_outside_temperature, DatasetMetadataType.float, 'outside_temperature')
# leap_binder.set_metadata(metadata_speed, DatasetMetadataType.float, 'speed')
# leap_binder.set_metadata(metadata_yaw_rate, DatasetMetadataType.float, 'yaw_rate')

# leap_binder.set_custom_layer(OnnxReduceMean, "OnnxReduceMean")
# leap_binder.set_custom_layer(OnnxSqrt, "OnnxSqrt")
# leap_binder.set_custom_layer(OnnxErf, "OnnxErf")


# leap_binder.add_custom_metric(loss_visualizer, 'loss_visualizer', LeapDataType.Image)

leap_binder.add_prediction('depth_estimation', ['1'])
