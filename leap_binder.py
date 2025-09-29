from typing import Dict, Union

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_input_encoder, tensorleap_gt_encoder, \
    tensorleap_metadata, tensorleap_custom_visualizer
from matplotlib import colors
from matplotlib import cm as cmx
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.contract.enums import LeapDataType

from GLPN.GLPN import IMG_PROCESSOR
from GLPN.configs import data_config, gcs_config
from tl_helpers.preprocess import subset_images
from utils.gcs_utils import _download
from eval.loss import si_log_loss, pixelwise_si_log_loss
from eval.metrics import calc_errors


# from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
# from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
# from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt


# ----------------------------------- Input ------------------------------------------
@tensorleap_input_encoder('normalized_image', channel_dim=1)
def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data.iloc[idx]['image']  # [idx % data["real_size"]]
    cloud_path = gcs_config.data_dir + cloud_path
    fpath = _download(cloud_path)
    img = Image.open(fpath)  # .convert('RGB')
    img = img.resize(data_config.image_size)
    img = IMG_PROCESSOR(img)  # , return_tensors="pt")
    return img['pixel_values'][0]


# ----------------------------------- GT ------------------------------------------
@tensorleap_gt_encoder('mask')
def gt_depth(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data.iloc[idx]['gt']  # [idx % data["real_size"]]
    cloud_path = gcs_config.data_dir + cloud_path
    fpath = _download(cloud_path)
    depth_png = Image.open(fpath).resize(data_config.image_size)
    depth_png = np.array(depth_png, dtype=int)

    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.
    return depth


# ----------------------------------- Metadata ------------------------------------------
@tensorleap_metadata('meta')
def all_metadata(idx, data: PreprocessResponse) -> Dict[str, Union[int, float, str]]:
    img = input_image(idx, data)
    depth = gt_depth(idx, data)
    metadata_arr = {
        'idx' : idx,
        'brightness': np.mean(img).astype(np.float32),
        'category': data.data.iloc[idx]['category'],
        'filenumber': int(data.data.iloc[idx]['image'].split('/')[-1].split('.png')[0]),
        'folder': data.data.iloc[idx]['folder'],
        'depth_mean': depth.mean(),
        'depth_std': depth.std(),
        'depth_min': depth.min(),
        'depth_max': depth.max()
    }
    return metadata_arr

# ---------------- Vis ------------------------

@tensorleap_custom_visualizer('depth_pred_vis', visualizer_type=LeapDataType.Image)
def depth_prediction_vis(pred) -> LeapImage:
    pred = pred[0, ...] if len(pred.shape) == 4 else pred
    output = np.squeeze(pred)
    formatted = (output * 255 / np.max(output)).astype("uint8")
    cmap = plt.get_cmap(data_config.cmap)
    colored_depth = cmap(formatted).astype(np.float32)[..., :-1]
    return LeapImage((colored_depth * 255).astype("uint8"))

@tensorleap_custom_visualizer('overlayed_depth_pred_vis', visualizer_type=LeapDataType.Image)
def overlayed_depth_prediction_vis(image, pred) -> LeapImage:
    image = np.squeeze(np.transpose(image, [0,2,3,1]))
    pred = np.squeeze(pred)
    # pred = tf.transpose(pred, perm=[1, 0])
    data = depth_prediction_vis(pred).data / 255.
    overlayed_image = ((data * 1 + image * 0.2).clip(0, 1) * 255).astype(np.uint8)
    return LeapImage(overlayed_image)

@tensorleap_custom_visualizer('overlayed_depth_gt_vis', visualizer_type=LeapDataType.Image)
def overlayed_depth_gt_vis(image, gt) -> LeapImage:
    image = np.squeeze(np.transpose(image, [0, 2, 3, 1]))
    image = np.squeeze(image)
    gt = np.squeeze(gt)
    data = depth_gt_vis(gt).data / 255.
    overlayed_image = ((data * 1 + image * 0.2).clip(0, 1) * 255).astype(np.uint8)
    return LeapImage(overlayed_image)

@tensorleap_custom_visualizer('depth_gt_vis', visualizer_type=LeapDataType.Image)
def depth_gt_vis(gt) -> LeapImage:
    gt = gt[0, ...] if len(gt.shape) == 3 else gt
    # Normalize depth values to the range [0, 1]
    normalized_depth = (gt - gt.min()) / (gt.max() - gt.min())
    cmap = plt.get_cmap(data_config.cmap)
    colored_depth = cmap(normalized_depth).astype(np.float32)[..., :-1]
    return LeapImage((colored_depth * 255).astype("uint8"))


def depth_loss(y_true, y_pred) -> LeapImage:
    data = pixelwise_si_log_loss(y_true, y_pred).numpy()
    if len(data.shape) == 4:
        data = data[0, ...]
    if len(data.shape) == 2:
        data = data[..., np.newaxis]
    return data

@tensorleap_custom_visualizer('depth_loss', visualizer_type=LeapDataType.Image)
def loss_visualizer(image, prediction, gt) -> LeapImage:
    image = np.squeeze(np.transpose(image, [0, 2, 3, 1]))
    prediction =  np.squeeze(prediction)
    gt = np.squeeze(gt)
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    ls_image = depth_loss(gt[None, ...], prediction)
    ls_image = ls_image.clip(0, np.percentile(ls_image, 95))
    ls_image /= ls_image.max()
    heatmap = scalarMap.to_rgba(ls_image[..., 0])[..., :-1]
    overlayed_image = ((heatmap * 0.6 + image * 0.4).clip(0, 1) * 255).astype(np.uint8)
    return LeapImage(overlayed_image)


# ----------------------------------- Binding ------------------------------------------


#leap_binder.set_preprocess(subset_images)

#leap_binder.set_input(input_image, 'normalized_image')
#leap_binder.set_ground_truth(gt_depth, 'mask')

#leap_binder.set_metadata(metadata_filenumber, 'filenumber')
#leap_binder.set_metadata(metadata_category, 'category')
#leap_binder.set_metadata(metadata_folder, 'folder')
#leap_binder.set_metadata(metadata_idx, 'idx')
#leap_binder.set_metadata(metadata_brightness, 'brightness')
#leap_binder.set_metadata(metadata_depth_mean, 'depth_mean')
#leap_binder.set_metadata(metadata_depth_std, 'depth_std')
#leap_binder.set_metadata(metadata_depth_min, 'depth_min')
#leap_binder.set_metadata(metadata_depth_max, 'depth_max')

# leap_binder.set_custom_layer(OnnxReduceMean, "OnnxReduceMean")
# leap_binder.set_custom_layer(OnnxSqrt, "OnnxSqrt")
# leap_binder.set_custom_layer(OnnxErf, "OnnxErf")

#leap_binder.set_visualizer(function=depth_prediction_vis, visualizer_type=LeapDataType.Image, name='depth_pred_vis')
# leap_binder.set_visualizer(function=overlayed_depth_prediction_vis, visualizer_type=LeapDataType.Image,
#                            name='overlayed_depth_pred_vis')
# leap_binder.set_visualizer(function=overlayed_depth_gt_vis, visualizer_type=LeapDataType.Image,
#                            name='overlayed_depth_gt_vis')
#leap_binder.set_visualizer(function=depth_gt_vis, visualizer_type=LeapDataType.Image, name='depth_gt_vis')
#leap_binder.set_visualizer(function=loss_visualizer, visualizer_type=LeapDataType.Image, name='depth_loss')

#leap_binder.add_custom_loss(si_log_loss, 'si_log_loss')
#leap_binder.add_custom_metric(calc_errors, 'error')

