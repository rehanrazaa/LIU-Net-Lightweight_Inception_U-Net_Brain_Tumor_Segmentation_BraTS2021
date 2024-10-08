# -*- coding: utf-8 -*-
"""metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UcIlEm3scgUczdvwBvvhY_-_P3goshOI

# Here is the code of all the metrices that are being used for the current experiemnts.
Please note few of the metrices are just given or might be not used. Thanks to https://github.com/shalabh147/Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks/blob/master/metrics.py for some metrices code.
"""

# Dice score
def _dice_hard_coe(target, output, smooth=1e-5):
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(output, target))
    l = tf.reduce_sum(output)
    r = tf.reduce_sum(target)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    return tf.reduce_mean(hard_dice)
# Dice score for Whole Tumor
def brats_wt(y_true, y_pred):
    # whole tumor
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    gt_wt = tf.cast(tf.identity(y_true), tf.int32)
    gt_wt = tf.where(tf.equal(2, gt_wt), 1 * tf.ones_like(gt_wt), gt_wt)  # ground_truth_wt[ground_truth_wt == 2] = 1
    gt_wt = tf.where(tf.equal(3, gt_wt), 1 * tf.ones_like(gt_wt), gt_wt)  # ground_truth_wt[ground_truth_wt == 3] = 1
    pd_wt = tf.cast(tf.round(tf.identity(y_pred)), tf.int32)
    pd_wt = tf.where(tf.equal(2, pd_wt), 1 * tf.ones_like(pd_wt), pd_wt)  # predictions_wt[predictions_wt == 2] = 1
    pd_wt = tf.where(tf.equal(3, pd_wt), 1 * tf.ones_like(pd_wt), pd_wt)  # predictions_wt[predictions_wt == 3] = 1
    return _dice_hard_coe(gt_wt, pd_wt)

# Dice score for Tumor Core
def brats_tc(y_true, y_pred):
    # tumor core
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    gt_tc = tf.cast(tf.identity(y_true), tf.int32)
    gt_tc = tf.where(tf.equal(2, gt_tc), 0 * tf.ones_like(gt_tc), gt_tc)  # ground_truth_tc[ground_truth_tc == 2] = 0
    gt_tc = tf.where(tf.equal(3, gt_tc), 1 * tf.ones_like(gt_tc), gt_tc)  # ground_truth_tc[ground_truth_tc == 3] = 1
    pd_tc = tf.cast(tf.round(tf.identity(y_pred)), tf.int32)
    pd_tc = tf.where(tf.equal(2, pd_tc), 0 * tf.ones_like(pd_tc), pd_tc)  # predictions_tc[predictions_tc == 2] = 0
    pd_tc = tf.where(tf.equal(3, pd_tc), 1 * tf.ones_like(pd_tc), pd_tc)  # predictions_tc[predictions_tc == 3] = 1
    return _dice_hard_coe(gt_tc, pd_tc)

# Dice score for Enhancing Tumor
def brats_et(y_true, y_pred):
    # enhancing tumor
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    gt_et = tf.cast(tf.identity(y_true), tf.int32)
    gt_et = tf.where(tf.equal(1, gt_et), 0 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 1] = 0
    gt_et = tf.where(tf.equal(2, gt_et), 0 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 2] = 0
    gt_et = tf.where(tf.equal(3, gt_et), 1 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 3] = 1
    pd_et = tf.cast(tf.round(tf.identity(y_pred)), tf.int32)
    pd_et = tf.where(tf.equal(1, pd_et), 0 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 1] = 0
    pd_et = tf.where(tf.equal(2, pd_et), 0 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 2] = 0
    pd_et = tf.where(tf.equal(3, pd_et), 1 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 3] = 1
    return _dice_hard_coe(gt_et, pd_et)

# Computing class wise Sensitivity and Specificity
# taken from: https://github.com/shalabh147/Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks/blob/master/metrics.py
def dice_coefficient(y_true, y_pred, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    """
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))

def compute_class_sens_spec(pred, label, class_num):
    """
    Compute sensitivity and specificity for a particular example
    for a given class.

    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (num classes, height, width, depth).
        label (np.array): binary array of labels, shape is
                          (num classes, height, width, depth).
        class_num (int): number between 0 - (num_classes -1) which says
                         which prediction class to compute statistics
                         for.

    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """

    # extract sub-array for specified class
    class_pred = pred[class_num]
    class_label = label[class_num]

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # compute true positives, false positives,
    # true negatives, false negatives
    tp = np.sum((class_pred == 1) & (class_label == 1))
    tn = np.sum((class_pred == 0) & (class_label == 1))
    fp = np.sum((class_pred == 1) & (class_label == 0))
    fn = np.sum((class_pred == 0) & (class_label == 0))

    # compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    ### END CODE HERE ###

    return sensitivity, specificity


def get_sens_spec_df(pred, label):
    patch_metrics = pd.DataFrame(
        columns = ['Nothing'
                    'Edema',
                   'Non-Enhancing Tumor',
                   'Enhancing Tumor'],
        index = ['Sensitivity',
                 'Specificity'])

    for i, class_name in enumerate(patch_metrics.columns):
        sens, spec = compute_class_sens_spec(pred, label, i)
        patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
        patch_metrics.loc['Specificity', class_name] = round(spec,4)

    return patch_metrics

# Classwise Sensitivity and Specificity

def sensitivity_wt(true_positive, false_negative):
    """
    Calculate sensitivity (true positive rate) given the true positive and false negative counts.
    """
    return true_positive / (true_positive + false_negative)


def specificity_wt(true_negative, false_positive):
    """
    Calculate specificity (true negative rate) given the true negative and false positive counts.
    """
    return true_negative / (true_negative + false_positive)

def sensitivity_tc(true_positive, false_negative):
    """
    Calculate sensitivity (true positive rate) given the true positive and false negative counts.
    """
    return true_positive / (true_positive + false_negative)


def specificity_tc(true_negative, false_positive):
    """
    Calculate specificity (true negative rate) given the true negative and false positive counts.
    """
    return true_negative / (true_negative + false_positive)

def sensitivity_et(true_positive, false_negative):
    """
    Calculate sensitivity (true positive rate) given the true positive and false negative counts.
    """
    return true_positive / (true_positive + false_negative)


def specificity_et(true_negative, false_positive):
    """
    Calculate specificity (true negative rate) given the true negative and false positive counts.
    """
    return true_negative / (true_negative + false_positive)

# Loss Functions
def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3),
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.
    """

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))

    return dice_loss