import time
import numpy as np
import tensorflow as tf
import os

from KS_lib.tf_model_he_cell_classification import tf_model_input_test
from KS_lib.tf_model_he_cell_classification import tf_model
from KS_lib.prepare_data import routine
from KS_lib.general import matlab
from itertools import izip

########################################################################################################################
def batch_processing(filename, coordinate, sess, logits_test, parameters, images_test, keep_prob,
                     mean_image, variance_image, flags, he_dcis_segmentation_path):
    # Read image and extract patches
    patches, patches_mask, image_size, nPatches = tf_model_input_test.read_data_test(filename, coordinate, flags, he_dcis_segmentation_path)

    def batches(generator, size):
        source = generator
        while True:
            chunk = [val for _, val in izip(xrange(size), source)]
            if not chunk:
                raise StopIteration
            yield chunk

    # Construct batch indices
    batch_index = range(0, nPatches, flags['test_batch_size'])
    if nPatches not in batch_index:
        batch_index.append(nPatches)

    # Process all_patches
    shape = np.hstack([nPatches, flags['size_output_patch']])
    shape[-1] = logits_test.get_shape()[3].value
    shape[1] = 1
    shape[2] = 1
    all_patches = np.zeros(shape, dtype=np.float32)
    all_patches[:,:,:,0] = 1.0

    # ncells = int(np.argmin([flags['test_batch_size']-np.mod(25*x,flags['test_batch_size']) for x in range(21)]))
    ncells = np.int(flags['test_batch_size']/1.0)
    for ipatch, chunk in enumerate(zip(batches(patches, ncells),
                                       batches(patches_mask, ncells))):

        # start_time = time.time()
        # start_idx = batch_index[ipatch]
        # end_idx = batch_index[ipatch + 1]

        idx = list()
        img = list()
        for icell in range(len(chunk[1])):
            tmp = list()
            for i in range(len(chunk[1][icell]['image'])):
                tmp.append(np.sum(chunk[1][icell]['image'][i] == 255.0) / float(chunk[1][icell]['image'][i].size))
            if np.sum(np.array(tmp) > 0.5) / len(tmp) > 0.5:
                idx.append(icell)
                img.append(chunk[0][icell]['image'])

        if img:
            img = np.vstack(img)

            subidx = range(0, img.shape[0], flags['test_batch_size'])
            if img.shape[0] not in subidx:
                subidx.append(img.shape[0])

            allpred = list()
            for times in range(len(subidx)-1):
                temp = tf_model_input_test.inputs_test(img[subidx[times]:subidx[times+1],:,:,:], mean_image, variance_image)

                if temp.shape[0] < flags['test_batch_size']:
                    rep = np.tile(temp[-1, :, :, :], [flags['test_batch_size'] - temp.shape[0], 1, 1, 1])
                    newtemp = np.vstack([temp, rep])
                    pred = sess.run(logits_test, feed_dict={images_test: newtemp, keep_prob: 1.0})
                    pred = pred[0:temp.shape[0],:,:,:]
                else:
                    pred = sess.run(logits_test, feed_dict={images_test: temp, keep_prob: 1.0})

                allpred.append(pred)

            allpred = np.vstack(allpred)
            allpred = np.vsplit(allpred,len(idx))
            allpred = [np.mean(pred,axis=0) for pred in allpred]

            for i, iidx in enumerate(idx):
                all_patches[ipatch*ncells + iidx, :, :, :] = allpred[i]

        # duration = time.time() - start_time
        # print('processing step %d/%d (%.2f sec/step)' % (ipatch + 1, nPatches/float(ncells), duration))

    result = all_patches
    result = np.reshape(result,(result.shape[0],result.shape[3]))

    return result

########################################################################################################################
def test(object_folder, model_path, filename_list, coordinate_list, flags,
         result_path, he_dcis_segmentation_path, igpu):
    checkpoint_dir = os.path.join(object_folder, 'checkpoint')
    mat_contents = matlab.load(os.path.join(checkpoint_dir, 'network_stats.mat'))
    mean_image = np.float32(mat_contents['mean_image'])
    variance_image = np.float32(mat_contents['variance_image'])

    ###########################################################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = igpu
    ###########################################################

    with tf.Graph().as_default(), tf.device(flags['gpu']):
        keep_prob = tf.placeholder(tf.float32)
        # Place holder for patches
        images_test = tf.placeholder(tf.float32, shape=(np.hstack([flags['test_batch_size'], flags['size_input_patch']])))
        # Network
        with tf.variable_scope("network") as scope:
            logits_test, parameters = tf_model.inference(images_test, keep_prob, flags)
        # Saver and initialisation
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags['gpu_memory_fraction'])
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config = config) as sess:
            # Initialise and load variables
            sess.run(init)
            saver.restore(sess, model_path)

            # result_dir = os.path.join(object_folder, flags['result_folder'])
            result_dir = result_path
            routine.create_dir(result_dir)

            for iImage, (file, coordinate) in enumerate(zip(filename_list,coordinate_list)):
                start_time = time.time()

                file = file[0]
                coordinate = coordinate[0]
                if os.path.exists(coordinate):
                    basename = os.path.basename(file)
                    basename = os.path.splitext(basename)[0]
                    savename = os.path.join(result_dir, basename + '.mat')
                    if not os.path.exists(savename):
                        # print('processing image %d/%d' % (iImage + 1, len(filename_list)))
                        result = batch_processing(file, coordinate, sess, logits_test, parameters, images_test,
                                              keep_prob, mean_image, variance_image, flags, he_dcis_segmentation_path)
                        matlab.save(savename,{'prediction':result})

                duration = time.time() - start_time
                print('Finish clasifying cells on the H&E image of sample %d out of %d samples (%.2f sec)' %
                      (iImage + 1, len(filename_list), duration))

########################################################################################################################
