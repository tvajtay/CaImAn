#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev

This pipeline demo has since been modified to take CLI arguments as part of a
SLURM script and save labeled images of the final detected components in the
origin directory for each respective movie in addition to the HDF5 file and
a AVI of the denoised components.
@tvajtay
"""

#%%
import numpy as np
import imageio
import os
import cv2
from os.path import isfile, join
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf import params
from pathlib import Path
import caiman as cm

#%%
#Need to search executed directory for files
fnames = sorted([f for f in os.listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) and f[-4:] in ['.tif','.sbx','.tf8','.btf']])
#%%
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


#%%
def main():
#%% First setup some parameters for data and motion correction

    # dataset dependent parameters
    fr = 15.49             # imaging rate in frames per second
    decay_time = 0.9    # length of a typical transient in seconds
    dxy = (1.4, 1.4)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # motion correction parameters
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy'
    }

    opts = params.CNMFParams(params_dict=mc_dict)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

# %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

# %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='_memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%  parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    gnb = 3                  # number of global background components
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 20                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    K = 3                    # number of components per patch
    gSig = [7, 7]            # expected half size of neurons in pixels
    method_init = 'greedy_roi'   # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 2                     # spatial subsampling during initialization
    tsub = 2                     # temporal subsampling during intialization
    s_min = -20                   # minimum signal amplitude needed in order for a transient to be considered as activity

    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub,
                 's_min': s_min}

    opts.change_params(params_dict=opts_dict)

# %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

# %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)


# %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm.params.change_params({'p': p})
    cnm2 = cnm.refit(images, dview=dview)
    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 2  # Overall minimum signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.15 # neurons with cnn probability lower than this value are rejected
    min_size_neuro = 0.1*gSig[0]*np.pi**2
    max_size_neuro = 2.5*gSig[0]*np.pi**2

    cnm2.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest,
                               'min_size_neuro': min_size_neuro,
                               'max_size_neuro': max_size_neuro,})

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    #%% update object with selected components
    cnm2.estimates.select_components(use_object=True)

    #%% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)


    #cnm2.estimates.threshold_spatial_components(maxthr=0.85)
    #cnm2.estimates.remove_small_large_neurons(min_size_neuro=min_size_neuro, max_size_neuro=max_size_neuro)

    # %% Generate heat image of video and save data
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    cnm2.estimates.Cn = Cn
    cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')


    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)

    #Create denoised video as matrix
    denoised = cm.movie(cnm2.estimates.A.dot(cnm2.estimates.C) + \
                    cnm2.estimates.b.dot(cnm2.estimates.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])

    #Normalizing denoised matrix to the [0,255] range prior to type conversion
    max_pixel = np.nanmax(denoised)
    norm_denoised = (denoised/max_pixel)*255

    #Type conversion so imageio doesn't get mad
    norm_denoised = np.uint8(norm_denoised)

    #Saving a denoised avi of the analyzed video
    sname = Path(args.in_file).stem + '.avi'
    imageio.mimwrite(sname, norm_denoised, fps=15.49)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
