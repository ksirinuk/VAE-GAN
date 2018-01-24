import os
import numpy as np
import re

from KS_lib.image import KSimage
from KS_lib.general import matlab

###############################################################
def generate_linear_gradient_for_merger(output_patch_size, stride):
    start_left = 0
    center_left = int(np.ceil(output_patch_size / 2.0) - 1)
    end_left = output_patch_size - 1

    start_right = stride
    center_right = int(stride + np.ceil(output_patch_size / 2.0) - 1)
    end_right = stride + output_patch_size - 1

    if stride >= output_patch_size:
        factor_left = np.ones(output_patch_size)
        factor_right = np.ones(output_patch_size)
    else:
        factor_left = np.zeros(output_patch_size)
        factor_left[start_left:center_left + 1] = 1

        if start_right > center_left:
            factor_left[center_left:start_right] = 1
            overlap_start = start_right
        else:
            overlap_start = center_left + 1

        if center_right <= end_left:
            factor_left[center_right:end_left+1] = 0
            overlap_end = center_right - 1
        else:
            overlap_end = end_left
        seq = np.arange(1,overlap_end - overlap_start + 2,1)
        factor_left[overlap_start:overlap_end+1] = seq[::-1] \
                                                   / float(overlap_end - overlap_start + 2)

        factor_right = np.zeros(end_right+1)
        factor_right[center_right:(end_right+1)] = 1
        if end_left < center_right:
            factor_right[end_left + 1:center_right+1] = 1
            overlap_end = end_left
        else:
            overlap_end = center_right - 1

        if start_right <= center_left:
            factor_right[start_right:center_left+1] = 0
            overlap_start = center_left + 1
        else:
            overlap_start = start_right

        factor_right[overlap_start:overlap_end+1] = np.arange(1,overlap_end - overlap_start + 2,1) \
                                                    / float(overlap_end - overlap_start + 2)


        factor_right = factor_right[start_right:end_right+1]

    return factor_left, factor_right

#####################################################################################
def MergePatches_test(patches, stride, image_size, sizeInputPatch, sizeOutputPatch, flags):
    patches = np.float32(patches)

    ntimes_row = int(np.floor((image_size[0] - sizeInputPatch[0]) / float(stride[0])) + 1)
    ntimes_col = int(np.floor((image_size[1] - sizeInputPatch[1]) / float(stride[1])) + 1)
    rowRange = range(0, ntimes_row * stride[0], stride[0])
    colRange = range(0, ntimes_col * stride[1], stride[1])

    displacement_row = int(round((sizeInputPatch[0] - sizeOutputPatch[0]) / 2.0))
    displacement_col = int(round((sizeInputPatch[1] - sizeOutputPatch[1]) / 2.0))

    image = np.zeros([image_size[0], image_size[1], patches.shape[3]], dtype=np.float32)

    factor_up_row, factor_down_row = generate_linear_gradient_for_merger(sizeOutputPatch[0], stride[0])
    factor_left_col, factor_right_col = generate_linear_gradient_for_merger(sizeOutputPatch[1], stride[1])

    factor_left_col = factor_left_col.reshape(1, len(factor_left_col), 1)
    factor_right_col = factor_right_col.reshape(1, len(factor_right_col), 1)
    factor_up_row = factor_up_row.reshape(len(factor_up_row), 1, 1)
    factor_down_row = factor_down_row.reshape(len(factor_down_row), 1, 1)

    factor_left_col = np.tile(factor_left_col, [sizeOutputPatch[0], 1, patches.shape[3]])
    factor_right_col = np.tile(factor_right_col, [sizeOutputPatch[0], 1, patches.shape[3]])
    factor_up_row = np.tile(factor_up_row, [1, image_size[1], patches.shape[3]])
    factor_down_row = np.tile(factor_down_row, [1, image_size[1], patches.shape[3]])

    ####################################################################################################################
    for index1, row in enumerate(rowRange):

        row_strip = np.zeros([sizeOutputPatch[0], image_size[1], patches.shape[3]], dtype=np.float32)

        for index2, col in enumerate(colRange):

            temp = patches[(index1 * len(colRange)) + index2, :, :, :]
            if index2 != 0:
                temp = temp * factor_right_col

            row_strip[:, col + displacement_col: col + displacement_col + sizeOutputPatch[1], :] += temp

            if index2 != len(colRange):
                row_strip[:, col + displacement_col : col + displacement_col + sizeOutputPatch[1],
                :] = row_strip[:, col + displacement_col : col + displacement_col + sizeOutputPatch[1],
                    :] * factor_left_col

        if index1 != 0:
            row_strip = row_strip * factor_down_row

        image[row + displacement_row: row + displacement_row + sizeOutputPatch[0], :, :] += row_strip

        if index1 != len(rowRange):
            image[row + displacement_row : row + displacement_row + sizeOutputPatch[0], :, :] = \
            image[ row + displacement_row : row + displacement_row + sizeOutputPatch[0], :, :] * factor_up_row


    ################################################################################################################


    image = image[flags['size_input_patch'][0]:image.shape[0] - flags['size_input_patch'][0],
            flags['size_input_patch'][1]:image.shape[1] - flags['size_input_patch'][1],
            :]
    return image

#####################################################################################
def ExtractPatches_test(sizeInputPatch, stride, image):
    ntimes_row = int(np.floor((image.shape[0] - sizeInputPatch[0])/float(stride[0])) + 1)
    ntimes_col = int(np.floor((image.shape[1] - sizeInputPatch[1])/float(stride[1])) + 1)
    rowRange = range(0, ntimes_row*stride[0], stride[0])
    colRange = range(0, ntimes_col*stride[1], stride[1])
    # nPatches = len(rowRange) * len(colRange)
    #
    #
    # patches = np.empty([nPatches, sizeInputPatch[0], sizeInputPatch[1], image.shape[2]],
    #                        dtype=image.dtype)

    for index1, row in enumerate(rowRange):
        for index2, col in enumerate(colRange):
            # patches[(index1 * len(colRange)) + index2, :, :, :] = image[row:row + sizeInputPatch[0],
            #                                                           col:col + sizeInputPatch[1], :]
            yield (image[row:row + sizeInputPatch[0],col:col + sizeInputPatch[1], :])

    # return patches

#####################################################################################
def extract_patch_coordinate(dict_obj, dict_patch_size, coordinate):
    list_keys = dict_obj.keys()

    coordinate = np.around(coordinate)
    coordinate = coordinate.astype(np.int32)

    # temp = []
    # for key in list_keys:
    #     temp.append(dict_patch_size[key][0:2])

    for index, (pointx, pointy) in enumerate(coordinate):

        dict_patches = dict()

        for key in list_keys:
            obj = dict_obj[key]
            size_input_patch = dict_patch_size[key]

            centre_index_row = int(round((size_input_patch[0] - 1) / 2.0))
            centre_index_col = int(round((size_input_patch[1] - 1) / 2.0))

            if obj.ndim == 2:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                           int(pointy - centre_index_col): int(pointy + centre_index_col + 1)]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1]]
            else:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                          int(pointy - centre_index_col): int(pointy + centre_index_col + 1), :]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1],:]

        yield dict_patches

#####################################################################################
def extract_patch_coordinate_ensemble(dict_obj, dict_patch_size, coordinate, flags):
    list_keys = dict_obj.keys()

    coordinate = np.around(coordinate)
    coordinate = coordinate.astype(np.int32)

    x = np.arange(-flags['jittering_radius_test'], flags['jittering_radius_test']+1)
    y = np.arange(-flags['jittering_radius_test'], flags['jittering_radius_test']+1)

    if np.mod(flags['jittering_radius_test'],2) == 1:
        x = x[1:len(x):2]
        y = y[1:len(y):2]
    else:
        x = x[0:len(x):2]
        y = y[0:len(y):2]

    xx, yy = np.meshgrid(x, y)
    z = np.sqrt(xx ** 2 + yy ** 2) <= flags['jittering_radius_test']
    jittering_idx = zip(xx[z], yy[z])
    n_patches = len(jittering_idx)

    for index, (pointx, pointy) in enumerate(coordinate):

        dict_patches = dict()

        for key in list_keys:
            obj = dict_obj[key]
            size_input_patch = dict_patch_size[key]

            centre_index_row = int(round((size_input_patch[0] - 1) / 2.0))
            centre_index_col = int(round((size_input_patch[1] - 1) / 2.0))

            temp = np.empty([n_patches, dict_patch_size[key][0], dict_patch_size[key][1], obj.shape[2]],
                               dtype=obj.dtype)
            if obj.ndim == 2:
                for index2, (jitter_col, jitter_row) in enumerate(jittering_idx):
                    row = pointx + jitter_row
                    col = pointy + jitter_col
                    temp[index2,:,:] = obj[int(row - centre_index_row): int(row + centre_index_row + 1),
                                           int(col - centre_index_col): int(col + centre_index_col + 1)]

                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1]]
            else:
                for index2, (jitter_col, jitter_row) in enumerate(jittering_idx):
                    row = pointx + jitter_row
                    col = pointy + jitter_col
                    temp[index2,:,:,:] = obj[int(row - centre_index_row): int(row + centre_index_row + 1),
                                          int(col - centre_index_col): int(col + centre_index_col + 1), :]

                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1],:]

        yield dict_patches

#####################################################################################
def read_data_test(filename, coordinate, flags, he_dcis_segmentation_path):

    image = KSimage.imread(filename)

    files = [f for f in os.listdir(he_dcis_segmentation_path)
             if os.path.isfile(os.path.join(he_dcis_segmentation_path, f))]

    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    pos = [m.start() for m in re.finditer('_', basename)]
    basename = basename[0:pos[3] + 1]

    basename = [x for x in files if basename in x][0]

    dcis_mask_file = os.path.join(he_dcis_segmentation_path, basename)
    if os.path.exists(dcis_mask_file):
        dcis_mask = KSimage.imread(dcis_mask_file)
    else:
        dcis_mask = np.ones(shape=(image.shape[0], image.shape[1])) * 255.0
        dcis_mask = dcis_mask.astype(np.uint8)

    image = KSimage.imresize(image, 0.5)
    dcis_mask = KSimage.imresize(dcis_mask, 0.5)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=3)

    if dcis_mask.ndim == 2:
        dcis_mask = np.expand_dims(dcis_mask, axis=3)

    padrow = flags['size_input_patch'][0]
    padcol = flags['size_input_patch'][1]

    image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')
    dcis_mask = np.lib.pad(dcis_mask, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

    mat_content = matlab.load(coordinate)
    coordinate = mat_content['coordinate']
    coordinate = coordinate.astype(np.int32)

    coordinate = (coordinate/2.0).astype(np.int32)

    shifted_coordinate = np.copy(coordinate)
    shifted_coordinate[:, 0] += padrow
    shifted_coordinate[:, 1] += padcol

    # extract patches
    dict_obj = {'image': image}
    dict_patch_size = {'image': flags['size_output_patch']}
    # patches = extract_patch_coordinate(dict_obj, dict_patch_size, shifted_coordinate)
    patches = extract_patch_coordinate_ensemble(dict_obj, dict_patch_size, shifted_coordinate, flags)

    dict_obj = {'image': dcis_mask}
    dict_patch_size = {'image': flags['size_output_patch']}
    patches_mask = extract_patch_coordinate_ensemble(dict_obj, dict_patch_size, shifted_coordinate, flags)

    # patches = ExtractPatches_test(flags['size_input_patch'], stride, image)

    nPatches = coordinate.shape[0]

    return patches, patches_mask, image.shape, nPatches

#####################################################################################
def process_image_test(patches, mean_image, variance_image):
    # Subtract off the mean and divide by the variance of the pixels.
    epsilon = 1e-6
    if mean_image.ndim == 2:
        mean_image = np.expand_dims(mean_image, axis = 3)
        variance_image = np.expand_dims(variance_image, axis = 3)

    for ipatch in range(patches.shape[0]):
        image = patches[ipatch, :, :, :]
        image = image - mean_image
        image = image / np.sqrt(variance_image + epsilon)
        patches[ipatch, :, :, :] = image

    # temp_patches = np.empty([patches.shape[0], 224, 224, patches.shape[3]], dtype=np.float32)
    # for ipatch in range(patches.shape[0]):
    #     image = patches[ipatch, :, :, :]
    #     image = image - mean_image
    #     image = image / np.sqrt(variance_image + epsilon)
    #     image = scipy.ndimage.interpolation.zoom(image, (0.5, 0.5, 1.0))
    #     temp_patches[ipatch, :, :, :] = image
    #
    # patches = temp_patches

    return patches

#####################################################################################
def inputs_test(patches, mean_image, variance_image):
    patches = np.float32(patches)
    patches = process_image_test(patches, mean_image, variance_image)
    return patches
