import matplotlib.pyplot as plt
import numpy as np

from GLPN.GLPN import get_model
from data.kitti import Kitti

from leap_binder import input_image, gt_depth
from tl_helpers.preprocess import subset_images
from eval.loss import si_log_loss

if __name__ == "__main__":
    ds = Kitti()
    train, val = subset_images()
    img = input_image(0, train)
    gt = gt_depth(0, train)

    # model = load_model('../models/GLPN_Kitti.h5')
    model = get_model()

    # Infer
    y = model(img['pixel_values'])
    y_pred = y.predicted_depth.detach().numpy()
    y_pred = np.swapaxes(np.swapaxes(y_pred, 0, 2), 0, 1)

    # Compute Loss
    loss = si_log_loss(gt, y_pred)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # estimator = GLPNDepthEstimation()
    # prediction = estimator.infer_raw_image(image)
    # plt.imshow(prediction, cmap="jet")
    # x = np.swapaxes(np.swapaxes(gt['pixel_values'][0], 0, 2), 0, 1)

    x = np.swapaxes(np.swapaxes(img['pixel_values'][0], 0, 2), 0, 1)
    plt.imshow(x)

    plt.imshow(y_pred)
    plt.imshow(gt)

    print('Done!')
