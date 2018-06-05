"""
Some of these functions are based on
http://deeplearning.net/tutorial/code/utils.py.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2016
"""

import numpy as np
import matplotlib.pyplot as plt


def scale_unit_interval(mat, eps=1e-8):
    """Scales all values in `mat` to be between 0 and 1."""
    mat = mat.copy()
    mat -= mat.min()
    mat *= 1.0 / (mat.max() + eps)
    return mat


def array_to_pixels(mat):
    """Convert the given array to pixel values after scaling."""
    mat = scale_unit_interval(mat)
    out_array = np.zeros(mat.shape, dtype="uint8")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            out_array[i, j] = mat[i, j] * 255
    return out_array


def tile_images(X, image_shape, tile_shape, tile_spacing=(1, 1),
        scale_rows_unit_interval=True):
    """
    Transform the 2-D matrix `X`, which has one flattened data instance or
    filter per row, into a matrix of pixel values with the data instances or
    filters layed out as tiles.

    Parameters
    ----------
    X : 2-D matrix or 4-D tensor
        The data to transform. If the tensor is given, the data from the
        last two dimensions are tiled.
    image_shape : (height, width)
        Each row is reshaped to this dimensionality.
    tile_shape : (n_rows, n_columns)
        Number of rows and columns to have in the output.
    scale_rows_unit_interval : bool
        Should each row be scaled to interval of [0, 1] before plotting

    Return
    ------
    out_array : matrix of type int
        Can be passed directly to `PIL.Image.fromarray`.
    """

    assert len(image_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    assert len(X.shape) == 2 or len(X.shape) == 4

    if len(X.shape) == 4:
        n_filters_out, n_channels_in, image_h, image_w = X.shape
        image_shape = image_h, image_w
        X = X.copy()
        X = X.reshape(n_filters_out*n_channels_in, image_h*image_w)

    # Dimensions
    image_h, image_w = image_shape
    spacing_h, spacing_w = tile_spacing
    n_tiles_h, n_tiles_w = tile_shape

    # Output dimensionality 
    out_shape = [0, 0]
    out_shape[0] = (image_h + spacing_h) * n_tiles_h - spacing_h
    out_shape[1] = (image_w + spacing_w) * n_tiles_w - spacing_w

    # Output matrix
    out_array = np.zeros(out_shape, dtype="uint8")

    # Lay out tiles
    for i_tile in xrange(n_tiles_h):
        for j_tile in xrange(n_tiles_w):
            cur_image = X[i_tile * n_tiles_w + j_tile].reshape(image_shape)
            if scale_rows_unit_interval:
                cur_image = scale_unit_interval(cur_image)
            i = i_tile * (image_h + spacing_h)
            j = j_tile * (image_w + spacing_w)
            out_array[i:i + image_h, j:j + image_w] = cur_image * 255

    return out_array


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)
