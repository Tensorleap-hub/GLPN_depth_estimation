import sys
from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf

from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt

from leap_binder import *
from eval.loss import si_log_loss, pixelwise_si_log_loss
from GLPN.configs import data_config

from code_loader.visualizers.default_visualizers import default_image_visualizer

sys.setrecursionlimit(10000)

if __name__ == "__main__":
    train, val = subset_images()
    res = metadata_folder(0, val)
    # load model
    fpath = "/Users/daniellebenbashat/Projects/depth_estimation/GLPN/models/GLPN_Kitti.h5"
    model = load_model(fpath, custom_objects={
        'OnnxErf': OnnxErf,
        'OnnxReduceMean': OnnxReduceMean,
        'OnnxSqrt': OnnxSqrt
    })

    sizes = [data_config.train_size, data_config.val_size]
    for sub in [train, val]:
        for i in tqdm(range(sub.length)):
            try:
                x = input_image(i, sub)
                # res = metadata_folder(i, sub)
                # res = metadata_depth_max(i, sub)
                # res = metadata_depth_min(i, sub)
                gt = gt_depth(i, sub)
                vis_res = default_image_visualizer(x)
                # vis_res = depth_vis(gt)
                pred = model(x[None, ...])
                pred = tf.transpose(pred, perm=[0, 2, 1])
                pred = tf.reshape(pred, shape=(1, 480, 640, 1))
                img = depth_prediction_vis(pred)
                out = overlayed_depth_gt_vis(x, gt)
                img = depth_gt_vis(gt[None, ...])
                out = overlayed_depth_prediction_vis(x, pred)
                loss = si_log_loss(gt[None, ...], pred)
                img = loss_visualizer(x, pred, gt[None, ...])
                erros = calc_errors(gt[None, ...], pred)
                loss = pixelwise_si_log_loss(gt, pred[0, ...])
                pixels_loss = pixelwise_si_log_loss(gt[None, ...], pred)
                img = depth_loss(gt[None, ...], pred)
                loss = si_log_loss(gt[None, ...], pred)
            except Exception as e:
                print(f'{i}: error {e}')

    for i in range(15):
        x = input_image(i, val)


