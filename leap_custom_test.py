import sys
from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf
from typing import List
import keras.backend as K
import os

import onnxruntime as rt

# from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
# from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
# from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt

from leap_binder import *
from eval.loss import si_log_loss, pixelwise_si_log_loss
from GLPN.configs import data_config

from code_loader.visualizers.default_visualizers import default_image_visualizer

sys.setrecursionlimit(10000)

from keras.layers import Layer
import tensorflow as tf


class OnnxErf(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = tf.math.erf(inputs)
        return x

class OnnxReduceMean(Layer):
    def __init__(self, axes: List[int], keepdims: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        tensor = K.mean(inputs, keepdims=self.keepdims, axis=self.axes)
        return tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "axes": self.axes,
            "keepdims": self.keepdims,
        })
        return config

class OnnxSqrt(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = tf.math.sqrt(inputs)
        return x


if __name__ == "__main__":
    train, val = subset_images()
    res = metadata_folder(0, val)
    # load model
    dir_path = os.path.dirname(os.path.abspath(__file__))
    fpath = "models/GLPN_Kitti.h5"
    model = load_model(os.path.join(dir_path, fpath), custom_objects={
        'OnnxErf': OnnxErf,
        'OnnxReduceMean': OnnxReduceMean,
        'OnnxSqrt': OnnxSqrt
    })

    sess = rt.InferenceSession(os.path.join(dir_path, "models/GLPN_Kitti.onnx"))
    input_name_1 = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[-1].name

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
                # y_pred = sess.run([label_name], {input_name_1: np.moveaxis(x[None, ...])})[0]

                # pred = tf.transpose(pred, perm=[0, 2, 1])
                # pred = tf.reshape(pred, shape=(1, 480, 640, 1))
                depth_prediction_vis_ = depth_prediction_vis(pred)
                overlayed_depth_gt_vis_ = overlayed_depth_gt_vis(x, gt)
                depth_gt_vis_ = depth_gt_vis(gt[None, ...])
                overlayed_depth_prediction_vis_ = overlayed_depth_prediction_vis(x, pred[0, ...])
                loss = si_log_loss(gt[None, ...], pred)
                loss_vis = loss_visualizer(x, pred[0, ...], gt[None, ...])
                erros = calc_errors(gt[None, ...], pred)
                depth_loss_ = depth_loss(gt[None, ...], pred[0, ...])
            except Exception as e:
                print(f'{i}: error {e}')

    for i in range(15):
        x = input_image(i, val)


