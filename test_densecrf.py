import cv2
import random
import numpy as np

from Pydensecrf import *
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def show_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_img_fw():
    img = cv2.imread('examples/im1.ppm')
    anno_rgb = cv2.imread('examples/anno1.ppm').astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + \
        (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labeling.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # And create a mapping back from the labels to 32bit integer colors.
    # But remove the all-0 black, that won't exist in the MAP!
    colors = colors[1:]
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - 1
    print(n_labels, " labels and \"unknown\" 0: ", set(labels.flat))

    # Example using the DenseCRF class and the util functions
    d = DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(
        sdims=(3, 3), shape=img.shape[:2]).astype(np.float32)
    d.addPairwiseEnergy(feats, PottsCompatibility(3),
                        KernelType.DIAG_KERNEL,
                        NormalizationType.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2).astype(np.float32)
    d.addPairwiseEnergy(feats, PottsCompatibility(10),
                        KernelType.DIAG_KERNEL,
                        NormalizationType.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.

    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the
    # image.
    MAP = colorize[MAP, :]
    print MAP.shape
    show_img(MAP.reshape(img.shape))


def test_img_bw():
    NIT_ = 5
    img = cv2.imread('examples/im1.ppm')
    anno_rgb = cv2.imread('examples/anno1.ppm').astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + \
        (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labeling = np.unique(anno_lbl, return_inverse=True)

    # And create a mapping back from the labels to 32bit integer colors.
    # But remove the all-0 black, that won't exist in the MAP!
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    # n_labels = len(set(labeling.flat)) - 1
    # print(n_labels, " labels and \"unknown\" 0: ", set(labeling.flat))

    # Example using the DenseCRF class and the util functions
    N = img.shape[0] * img.shape[1]
    M = 4

    d = DenseCRF(img.shape[1] * img.shape[0], M)

    logistic_feature = np.ones((4, N), dtype=np.float32)
    logistic_transform = np.zeros((M, 4), dtype=np.float32)

    img_reshape = img.reshape(-1, 3)
    logistic_feature[:3, :] = img_reshape.T / 255.

    for j in xrange(logistic_transform.shape[1]):
        for i in xrange(logistic_transform.shape[0]):
            logistic_transform[i, j] = 0.01 * (1 - 2 * random.random())

    # U = unary_from_labels(labeling, n_labels, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(logistic_transform, logistic_feature)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(
        sdims=(3, 3), shape=img.shape[:2]).astype(np.float32)
    d.addPairwiseEnergy(feats, PottsCompatibility(1),
                        KernelType.DIAG_KERNEL,
                        NormalizationType.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2).astype(np.float32)
    d.addPairwiseEnergy(feats, PottsCompatibility(1),
                        KernelType.DIAG_KERNEL,
                        NormalizationType.NORMALIZE_SYMMETRIC)

    objective = IntersectionOverUnion(labeling.astype(np.int16))
    #objective = Hamming(labeling.astype(np.int32),np.ones(n_label).astype(np.float32))

    # Optimize the CRF in 3 phases:
    #  * First unary only
    #  * Unary and pairwise
    #  * Full CRF
    learning_params = np.array([[True, False, False],
                                [True, True, False],
                                [True, True, True]])

    for i in range(learning_params.shape[0]):
        # Setup the energy
        energy = CRFEnergy(d, objective, NIT_, learning_params[
                           i, 0], learning_params[i, 1], learning_params[i, 2])
        energy.setL2Norm(1e-3)

        # Minimize the energy
        p = minimizeLBFGS(energy, 2, True)

        # Save the values
        idx = 0
        if learning_params[i, 0]:
            print "HERE"
            print d.unaryParameters().shape
            d.setUnaryParameters(p[idx:idx + d.unaryParameters().shape[0]])
            idx += d.unaryParameters().shape[0]

        if learning_params[i, 1]:
            d.setLabelCompatibilityParameters(
                p[idx:idx + d.labelCompatibilityParameters().shape[0]])
            idx += d.labelCompatibilityParameters().shape[0]

        if learning_params[i, 2]:
            d.setKernelParameters(p[idx:idx + d.kernelParameters().shape[0]])
            idx += d.kernelParameters().shape[0]

    print "Pairwise Parameters: ", d.labelCompatibilityParameters()
    print "Kernel Parameters: ", d.kernelParameters()

    # Run five inference steps.
    Q = d.inference(NIT_)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labeling) back to the corresponding colors and save the
    # image.
    MAP = colorize[MAP, :]

    from ipdb import set_trace; set_trace()
    show_img(MAP.reshape(img.shape))


def test_img_bw_2D():
    NIT_ = 5
    img = cv2.imread('examples/im1.ppm')
    anno_rgb = cv2.imread('examples/anno1.ppm').astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + \
        (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)
    # And create a mapping back from the labels to 32bit integer colors.
    # But remove the all-0 black, that won't exist in the MAP!
    colors = colors[0:]
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".

    # Example using the DenseCRF class and the util functions
    d = DenseCRF2D(img.shape[1], img.shape[0], 4)  # n_labels)

    N = img.shape[0] * img.shape[1]
    M = 4  # n_labels
    logistic_feature = np.ones((4, N), dtype=np.float32)
    logistic_transform = np.zeros((M, 4), dtype=np.float32)

    img_reshape = img.reshape(-1, 3)
    logistic_feature[:3, :] = img_reshape.T / 255.

    for j in xrange(logistic_transform.shape[1]):
        for i in xrange(logistic_transform.shape[0]):
            logistic_transform[i, j] = 0.01 * (1 - 2 * random.random())

    # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(logistic_transform, logistic_feature)

    # This creates the color-independent features and then add them to the CRF
    d.addPairwiseGaussian(3, 3, PottsCompatibility(1), KernelType.DIAG_KERNEL,
                          NormalizationType.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    matrix = np.ones((M,M)).astype(np.float32)

    d.addPairwiseBilateral(80., 80., 13., 13., 13., img,
                           MatrixCompatibility(matrix),
                           KernelType.DIAG_KERNEL,
                           NormalizationType.NORMALIZE_SYMMETRIC)

    objective = IntersectionOverUnion(labeling.astype(np.int16))

    # Optimize the CRF in 3 phases:
    #  * First unary only
    #  * Unary and pairwise
    #  * Full CRF
    learning_params = np.array([[True, False, False],
                                [True, True, False],
                                [True, True, True]])

    for i in range(learning_params.shape[0]):
        # Setup the energy
        energy = CRFEnergy(d, objective, NIT_, learning_params[i, 0],
                           learning_params[i, 1], learning_params[i, 2])
        energy.setL2Norm(1e-3)

        # Minimize the energy
        p = minimizeLBFGS(energy, 2, True)

        # Save the values
        idx = 0
        if learning_params[i, 0]:
            d.setUnaryParameters(p[idx:idx + d.unaryParameters().shape[0]])
            idx += d.unaryParameters().shape[0]

        if learning_params[i, 1]:
            d.setLabelCompatibilityParameters(
                p[idx:idx + d.labelCompatibilityParameters().shape[0]])
            idx += d.labelCompatibilityParameters().shape[0]

        if learning_params[i, 2]:
            d.setKernelParameters(p[idx:idx + d.kernelParameters().shape[0]])
            idx += d.kernelParameters().shape[0]

    # Return parameters
    print "Unary Parameters: ", d.unaryParameters()
    print "Pairwise Parameters: ", d.labelCompatibilityParameters()
    print "Kernel Parameters: ", d.kernelParameters()

    # Run five inference steps.
    Q = d.inference(NIT_)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labeling) back to the corresponding colors and save the
    # image.
    MAP = colorize[MAP, :]

    show_img(MAP.reshape(img.shape))


if __name__ == '__main__':
    test_img_fw()
    test_img_bw()
    # test_img_bw_2D()
