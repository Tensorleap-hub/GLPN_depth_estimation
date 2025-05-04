from typing import Dict, Union, Any

import numpy as np
import tensorflow as tf


def calc_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Union[int, Any]]:
    # print(f"gt shape{gt.shape}, pred shape:{pred.shape}")
    pred = tf.transpose(pred, perm=[0,2,1])
    # pred = tf.squeeze(pred, axis=-1)
    valid_mask = tf.cast(gt > 0, dtype=tf.bool)
    gt = gt[:, valid_mask[0, ...]].astype(np.float32)#.reshape((gt.shape[0], -1))
    pred = pred.numpy()[:, valid_mask[0, ...]].astype(np.float32)#.reshape((pred.shape[0], -1))

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean(axis=-1).astype(np.float32)
    d2 = (thresh < 1.25 ** 2).mean(axis=-1).astype(np.float32)
    d3 = (thresh < 1.25 ** 3).mean(axis=-1).astype(np.float32)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean(axis=-1)).astype(np.float32)

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean(axis=-1))

    abs_rel = np.mean(np.abs(gt - pred) / gt, axis=-1)
    sq_rel = np.mean(((gt - pred)**2) / gt, axis=-1)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2, axis=-1) - np.mean(err, axis=-1) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err, axis=-1).astype(np.float32)

    dic = dict(silog=silog, log10=log10, abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log, d1=d1, d2=d2, d3=d3)
    return dic