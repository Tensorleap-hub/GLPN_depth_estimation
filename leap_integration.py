import sys

from code_loader.contract.datasetclasses import PredictionTypeHandler
from keras.models import load_model
import os
from leap_binder import *
from eval.loss import si_log_loss, pixelwise_si_log_loss
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, integration_test
import onnxruntime
sys.setrecursionlimit(10000)

prediction_type1 = PredictionTypeHandler('depth', ['high', 'low'], channel_dim=1)

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'models/GLPN_Kitti.onnx'
    sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))
    return sess

@integration_test()
def integration_test(idx, subset):
    plot_vis = True
    sess = load_model()
    input_name_1 = sess.get_inputs()[0].name

    # inputs and GT
    x = input_image(idx, subset)
    gt = gt_depth(idx, subset)

    # model
    pred = sess.run(None, {input_name_1: x})[0]

    # metrics
    errors = calc_errors(gt, pred)
    loss = si_log_loss(gt, pred)

    #vis
    depth_prediction_vis_ = depth_prediction_vis(pred)
    overlay_depth_prediction_vis_ = overlayed_depth_prediction_vis(x, pred)
    overlay_depth_gt_vis_ = overlayed_depth_gt_vis(x, gt)
    depth_gt_vis_ = depth_gt_vis(gt)
    loss_vis = loss_visualizer(x, pred, gt)

    if plot_vis:
        visualize(depth_prediction_vis_)
        visualize(overlay_depth_prediction_vis_)
        visualize(overlay_depth_gt_vis_)
        visualize(depth_gt_vis_)
        visualize(loss_vis)
    print(all_metadata(idx, subset))

if __name__ == "__main__":
    train, val = subset_images()
    integration_test(0, train)


