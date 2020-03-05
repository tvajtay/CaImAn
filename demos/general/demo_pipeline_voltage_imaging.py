#!/usr/bin/env python
"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.

Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).

"""

import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import h5py

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.source_extraction.volpy.mrcnn import visualize, neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
from caiman.paths import caiman_datadir

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

    # %%  Load demo movie and ROIs
    fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
    path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)

    # %% Setup some parameters for data and motion correction
    # dataset parameters
    fr = 400                                        # sample rate of the movie
    ROIs = None                                     # Region of interests
    index = None                                    # index of neurons
    weights = None                                  # reuse spatial weights by 
                                                    # opts.change_params(params_dict={'weights':vpy.estimates['weights']})
    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'index': index,
        'ROIs': ROIs,
        'weights': weights,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)

    # %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # %% start a cluster for parallel processing

    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # %%% MOTION CORRECTION
    # Create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run piecewise rigid motion correction
    mc.motion_correct(save_movie=True)
    dview.terminate()

    # %% motion correction compared with original movie
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # % movie subtracted from the mean
        m_orig2 = (m_orig - np.mean(m_orig, axis=0))
        m_rig2 = (m_rig - np.mean(m_rig, axis=0))
        moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
                                       m_rig2.resize(1, 1, ds_ratio)], axis=2)
        moviehandle1.play(fr=60, q_max=99.5, magnification=2)

   # %% Memory Mapping
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                               add_to_mov=border_to_0, dview=dview, n_chunks=10)

    dview.terminate()

    # %% change fnames to the new motion corrected one
    opts.change_params(params_dict={'fnames': fname_new})

    # %% SEGMENTATION
    # Create mean and correlation image
    use_maskrcnn = True  # set to True to predict the ROIs using the mask R-CNN
    if not use_maskrcnn:                 # use manual annotations
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  # load ROIs
        opts.change_params(params_dict={'ROIs': ROIs,
                                        'index': list(range(ROIs.shape[0])),
                                        'method': 'SpikePursuit'})
    else:
        m = cm.load(mc.mmap_file[0], subindices=slice(0, 20000))
        m.fr = fr
        img = m.mean(axis=0)
        img = (img-np.mean(img))/np.std(img)
        m1 = m.computeDFF(secsWindow=1, in_place=True)[0]
        m = m - m1
        Cn = m.local_correlations(swap_dim=False, eight_neighbours=True)
        img_corr = (Cn-np.mean(Cn))/np.std(Cn)
        summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32)
        del m
        del m1

        # %%
        # Mask R-CNN
        config = neurons.NeuronsConfig()

        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            IMAGE_RESIZE_MODE = "pad64"
            IMAGE_MAX_DIM = 512
            RPN_NMS_THRESHOLD = 0.7
            POST_NMS_ROIS_INFERENCE = 1000

        config = InferenceConfig()
        config.display()
        model_dir = os.path.join(caiman_datadir(), 'model')
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                      config=config)
        weights_path = download_model('mask_rcnn')
        model.load_weights(weights_path, by_name=True)
        results = model.detect([summary_image], verbose=1)
        r = results[0]
        ROIs_mrcnn = r['masks'].transpose([2, 0, 1])

    # %% visualize the result
        display_result = False

        if display_result:
            _, ax = plt.subplots(1,1, figsize=(16,16))
            visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                                    ['BG', 'neurons'], r['scores'], ax=ax,
                                    title="Predictions")

    # %% set rois
        opts.change_params(params_dict={'ROIs':ROIs_mrcnn,
                                        'index':list(range(ROIs_mrcnn.shape[0])),
                                        'method':'SpikePursuit'})

    # %% Trace Denoising and Spike Extraction
    c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)

    # %% some visualization
    print(np.where(vpy.estimates['passedLocalityTest'])[0])    # neurons that pass locality test
    n = 0
    
    # Processed signal and spikes of neurons
    plt.figure()
    plt.plot(vpy.estimates['trace'][n])
    plt.plot(vpy.estimates['spikeTimes'][n],
             np.max(vpy.estimates['trace'][n]) * np.ones(vpy.estimates['spikeTimes'][n].shape),
             color='g', marker='o', fillstyle='none', linestyle='none')
    plt.title('signal and spike times')
    plt.show()

    # Location of neurons by Mask R-CNN or manual annotation
    plt.figure()
    if use_maskrcnn:
        plt.imshow(ROIs_mrcnn[n])
    else:
        plt.imshow(ROIs[n])
    mv = cm.load(fname_new)
    plt.imshow(mv.mean(axis=0),alpha=0.5)
    
    # Spatial filter created by algorithm
    plt.figure()
    plt.imshow(vpy.estimates['spatialFilter'][n])
    plt.colorbar()
    plt.title('spatial filter')
    plt.show()
    
    

    # %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)


# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
