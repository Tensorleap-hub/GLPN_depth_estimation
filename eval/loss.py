import tensorflow as tf
import numpy as np
from scipy.optimize import minimize


def origin_si_log_loss(y_true, y_pred, epsilon=1e-7):
    y_pred = tf.squeeze(y_pred, axis=-1)
    # y_pred = tf.transpose(y_pred, perm=[0, 3, 2, 1])
    valid_mask = tf.cast(y_true > 0, dtype=tf.bool)

    # Add epsilon to avoid division by zero
    y_true = tf.where(valid_mask, y_true, epsilon)
    y_pred = tf.where(valid_mask, y_pred, epsilon)

    diff_log = tf.math.log(y_true[valid_mask].astype(np.float32)) - tf.math.log(y_pred[valid_mask].astype(np.float32))
    loss = tf.sqrt(tf.math.reduce_mean(tf.math.pow(diff_log, 2)) - 0.5 * tf.math.pow(tf.math.reduce_mean(diff_log), 2))

    return loss


def si_log_loss(y_true: np.ndarray, y_pred: np.ndarray) ->np.ndarray:
    epsilon = 1e-7
    # y_pred = tf.squeeze(y_pred, axis=-1)
    y_pred = tf.transpose(y_pred, perm=[0, 2, 1])
    valid_mask = tf.cast(y_true > 0, dtype=tf.bool)

    # Add epsilon to avoid division by zero
    y_true = tf.where(valid_mask, y_true, epsilon)
    y_pred = tf.where(valid_mask, y_pred, epsilon)

    diff_log = tf.math.log(y_true.astype(np.float32)) - tf.math.log(y_pred.astype(np.float32))
    n = valid_mask.size//len(valid_mask)
    loss = tf.sqrt(tf.math.divide(tf.math.reduce_sum(tf.math.pow(diff_log, 2), axis=(1, 2)), n) - 0.5 * tf.math.pow(tf.math.divide(tf.math.reduce_sum(diff_log, axis=(1, 2)), n), 2))
    return loss.numpy()


def old_pixelwise_si_log_loss(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=-1)
    valid_mask = tf.cast(y_true > 0, dtype=tf.bool)

    # Calculate the logarithms of target and predicted values
    log_true = tf.math.log(tf.maximum(y_true, 1e-7))  # Avoid taking the log of zero
    log_pred = tf.math.log(tf.maximum(y_pred, 1e-7))  # Avoid taking the log of zero

    # Calculate the squared difference of logarithms
    diff_log = log_true - log_pred
    squared_diff = tf.math.square(diff_log)

    # Calculate the mean squared difference
    mean_squared_diff = tf.reduce_mean(squared_diff, axis=-1)  # Calculate along the last dimension

    # Calculate the SiLogLoss
    silog_loss = tf.sqrt(mean_squared_diff - 0.5 * tf.math.square(tf.reduce_mean(diff_log, axis=-1)))

    # Add mask where there isn't valid GT
    silog_loss = tf.where(valid_mask, tf.expand_dims(silog_loss, -1), -1)
    return silog_loss


def pixelwise_si_log_loss(y_true, y_pred):
    # y_pred = tf.squeeze(y_pred, axis=-1)
    y_pred = tf.transpose(y_pred, perm=[1, 0])
    valid_mask = tf.cast(y_true > 0, dtype=tf.bool)

    # Define the objective function
    def objective_function(x):
        log_true = tf.math.log(tf.maximum(y_true, 1e-7))  # Avoid taking the log of zero
        log_pred = tf.math.log(tf.maximum(x*y_pred, 1e-7))  # Avoid taking the log of zero
        diff_log = (log_true - log_pred)
        n = valid_mask.size // len(valid_mask)
        loss_per_sample = tf.sqrt(tf.math.divide(tf.math.reduce_sum(tf.math.pow(diff_log, 2), axis=(0, 1)), n) - 0.5 * tf.math.pow(tf.math.divide(tf.math.reduce_sum(diff_log, axis=(0, 1)), n), 2))
        loss = tf.math.reduce_mean(loss_per_sample)
        # Calculate the logarithms of target and predicted values
        return loss

    # Initial guess for the parameter
    initial_guess = 1.

    # Minimize the objective function
    result = minimize(objective_function, initial_guess, method='BFGS')

    # Extract the optimized parameter value
    x = result.x[0]
    y_pred *= x

    log_true = tf.math.log(tf.maximum(y_true, 1e-7))  # Avoid taking the log of zero
    log_pred = tf.math.log(tf.maximum(x * y_pred, 1e-7))  # Avoid taking the log of zero
    error = tf.pow(log_true - log_pred, 2)
    error = tf.where(valid_mask, error, -1)

    return tf.expand_dims(error, -1).astype(float)
