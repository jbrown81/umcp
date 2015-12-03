#!/usr/bin/env python

#    This program is part of the UCLA Multimodal Connectivity Package (UMCP)
#
#    UMCP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2013 Jesse Brown

# pyentropy module is required for mutual information calculations
# http://code.google.com/p/pyentropy/

import os
import sys
import numpy as np
import nibabel as nib
import core

def vox_ts_corrs(nifti_file,coord=None,covariate_file=None,outnii_filename=False,mask_thresh=0,scrub_trs_file=None):
    """
    Take either: 1) an (x,y,z) coordinate or 2) an external covariate file (column) and calculate the correlation of that coordinate's timeseries
    with all other timeseries'
    """
    input = nib.load(nifti_file)
    input_d = input.get_data()
    if scrub_trs_file:
        scrub_trs = np.array(core.file_reader(scrub_trs_file)) # array of TRs to exclude
        keep_trs = np.nonzero(scrub_trs==0)[0] # array of TRs to include
        input_d = input_d[:, :, :, keep_trs]
    nonzero_coords = core.get_nonzero_coords(nifti_file,mask_thresh)
    ts_array = np.array([input_d[nz_coord[0],nz_coord[1],nz_coord[2], :] for nz_coord in nonzero_coords])
    if coord:
        nonzero_index = nonzero_coords.index(coord)
        seed_ts = ts_array[nonzero_index]
    else:
        seed_ts = core.file_reader(covariate_file)
        seed_ts = [item for sublist in seed_ts for item in sublist] # flatten list
    indiv_corr = [np.corrcoef([seed_ts,ts])[1][0] for ts in ts_array]
    coord_corrs = zip(nonzero_coords,indiv_corr)
    vox_corrs_image = np.zeros((input.shape[0:3]))
    
    for out_coord,out_corr in coord_corrs:
        vox_corrs_image[out_coord[0],out_coord[1],out_coord[2]]=out_corr
    if not outnii_filename:
        return vox_corrs_image
    else:
        outnifti = nib.Nifti1Image(vox_corrs_image, input.get_header().get_best_affine())
        outnifti.to_filename(outnii_filename)

def mask_ts_coors(nifti_file, mask, outnii_filename=None, mask_thresh=0,scrub_trs_file=None):
    """
    Calculates correlations between a mask's mean timeseries and all other voxels
    """
    input = nib.load(nifti_file)
    input_d = input.get_data()
    if scrub_trs_file:
        scrub_trs = np.array(core.file_reader(scrub_trs_file)) # array of TRs to exclude
        keep_trs = np.nonzero(scrub_trs==0)[0] # array of TRs to include
        input_d = input_d[:, :, :, keep_trs]
    nonzero_coords = core.get_nonzero_coords(nifti_file,mask_thresh)
    mask_coords = core.get_nonzero_coords(mask,mask_thresh)
    mask_array = [input_d[mask_coord[0], mask_coord[1], mask_coord[2], :] for mask_coord in mask_coords]
    mask_mean_ts = np.mean(mask_array,axis=0)
    del mask_array
    ts_array = np.array([input_d[nz_coord[0],nz_coord[1],nz_coord[2], :] for nz_coord in nonzero_coords])
    del input_d
    indiv_corr = [np.corrcoef([mask_mean_ts,ts])[1][0] for ts in ts_array]
    coord_corrs = zip(nonzero_coords,indiv_corr)
    xsize, ysize, zsize = input.shape[0:3]
    mask_corrs_image = np.zeros((xsize, ysize, zsize))
    for out_coord, out_corr in coord_corrs:
        mask_corrs_image[out_coord[0],out_coord[1],out_coord[2]] = out_corr
    if not outnii_filename:
        return mask_corrs_image
    else:
        outnifti = nib.Nifti1Image(mask_corrs_image, input.get_header().get_best_affine())
        outnifti.to_filename(outnii_filename)

def mask_mutualinfo_matrix(nifti_file,masks,outfile,mask_thresh=0,nbins=10):
    """
    Calculates mutual information matrix for a set of mask mean timeseries'
    """
    from pyentropy import DiscreteSystem
    mutualinfo_mat = np.zeros((len(masks),len(masks)))
    input = nib.load(nifti_file)
    input_d = input.get_data()
    if len(input.shape) > 3:
        ts_length = input.shape[3]
    else:
        ts_length = 1
    mean_bin_ts_array = np.zeros((len(masks),ts_length),dtype='int')
    for count,mask in enumerate(masks):
        mask_coords = core.get_nonzero_coords(mask,mask_thresh)
        mask_array = [input_d[mask_coord[0], mask_coord[1], mask_coord[2], :] for mask_coord in mask_coords]
        mean_ts = np.mean(mask_array,axis=0)
        l = np.linspace(min(mean_ts),max(mean_ts),nbins)
        mean_bin_ts_array[count,:] = np.digitize(mean_ts,l)-1 # set range to start at 0
        if count > 0:
            for prev in range(count):
                sys = DiscreteSystem(mean_bin_ts_array[count],(1,nbins),mean_bin_ts_array[prev],(1,nbins))
                sys.calculate_entropies(method='qe',calc=['HX','HXY','HiXY','HshXY'])
                mutualinfo_mat[count,prev] = sys.I()
    mutualinfo_mat_sym = core.symmetrize_mat(mutualinfo_mat,'bottom')
    np.savetxt('%s.txt'%outfile,mutualinfo_mat_sym)
    return mutualinfo_mat_sym

def mask_funcconnec_matrix(nifti_file,masks_files,outfile=None,masks_threshes = [],
                           multi_labels=[],partial=False,cov=False,zero_diag=True,
                           scrub_trs_file=None,pca=False,ts_outfile=None,covariate_ts_file=None):
    """
    Calculates correlation/covariance matrix for a set of mask mean timeseries'
    masks_files: list of mask filenames with full path, can either be one mask
                 per file (in which case multi_labels should be []) or one file
                 with multiple numerical labels (multi_labels = [num1, num2, ...])
    masks_threshes: list of numerical values to use as lower threshold for separate
                    mask files
    covariate_ts_file: text file with timeseries for nuisance covariates to partial out
    output options:
    1) correlation matrix
    2) partial correlation matrix
    3) covariance matrix
    """
    if multi_labels:
        masks_coords = core.get_mask_labels(masks_files[0], labels=multi_labels)
    else:
        if masks_threshes:
            masks_coords = []
            for count, mask in enumerate(masks_files):
                masks_coords.append(core.get_nonzero_coords(mask, masks_threshes(count)))
        else:
            masks_coords = [core.get_nonzero_coords(mask) for mask in masks_files]
    n_regions = len(masks_coords)
    connect_mat = np.zeros((len(masks_coords), len(masks_coords)))
    input = nib.load(nifti_file)
    input_d = input.get_data()
    if scrub_trs_file:
        scrub_trs = np.array(core.file_reader(scrub_trs_file)) # array of TRs to exclude
        keep_trs = np.nonzero(scrub_trs==0)[0] # array of TRs to include
    if len(input.shape) > 3:
        if scrub_trs_file:
            ts_length = len(keep_trs)
        else:
            ts_length = input.shape[3]
    else:
        ts_length = 1
    masks_mean_ts_array = np.zeros((len(masks_coords), ts_length))
    for count, mask_coords in enumerate(masks_coords):
        if scrub_trs_file:
            mask_array = [input_d[mask_coord[0], mask_coord[1], mask_coord[2], keep_trs] for mask_coord in mask_coords]
        else:
            mask_array = [input_d[mask_coord[0], mask_coord[1], mask_coord[2], :] for mask_coord in mask_coords]
        if pca:
            [coeff,score,latent] = princomp(np.matrix(mask_array))
            masks_mean_ts_array[count, :] = score[0,:]
        else:
            masks_mean_ts_array[count, :] = np.mean(mask_array, axis=0)
    if partial:
        mat = core.partialcorr_matrix(masks_mean_ts_array)
    elif cov:
        mat = np.cov(masks_mean_ts_array)
    elif covariate_ts_file:
        nuis_reg = np.array(core.file_reader(covariate_ts_file))
        mat = np.zeros((n_regions,n_regions))
        for i in range(n_regions):
          for j in range(i+1,n_regions):
            n1n2 = np.hstack((np.atleast_2d(masks_mean_ts_array[i,:]).T,np.atleast_2d(masks_mean_ts_array[j,:]).T))
            X = np.vstack((n1n2.T,nuis_reg.T))
            try:
                pc_mat = core.partialcorr_matrix(X)
                mat[i,j] = pc_mat[0,1]
            except:
                mat[i,j] = 0
                print('Mask %d is empty, correlation will be stored as 0'%(j))
        mat = mat + mat.T
    else:
        mat = np.corrcoef(masks_mean_ts_array)
    if zero_diag:
        mat = mat * abs(1-np.eye(mat.shape[0])) # zero matrix diagonal
    if outfile:
        np.savetxt('%s.txt'%outfile, mat)
    if ts_outfile:
        np.savetxt('%s.txt'%ts_outfile, masks_mean_ts_array)
    return mat, masks_mean_ts_array

def mask_variance(nifti_file, masks_file, outfile, std=False, scrub_trs_file=None, mask_thresh=0):
    """
    Calculates variance/standard deviation for a set of masks
    Takes a 4D BOLD nifti_file, 4D masks_file
    """
    input = nib.load(nifti_file)
    input_d = input.get_data()
    masks_input = nib.load(masks_file)
    masks_d = masks_input.get_data()
    num_masks = masks_d.shape[3]
    if scrub_trs_file:
        scrub_trs = np.array(core.file_reader(scrub_trs_file)) # array of TRs to exclude
        keep_trs = np.nonzero(scrub_trs==0)[0] # array of TRs to include
        input_d = input_d[:, :, :, keep_trs]
    ts_length = input_d.shape[3]
    ts_mat = np.zeros((num_masks,ts_length))
    var_mat = np.zeros((num_masks))
    for count in range(num_masks):
        mask_coords = np.nonzero(masks_d[:,:,:,count])
        mask_coords = zip(*mask_coords)
        mask_array = [input_d[mask_coord[0], mask_coord[1], mask_coord[2], :] for mask_coord in mask_coords]
        ts_mat[count,:] = np.mean(mask_array, axis=0)
        if std:
            var_mat[count] = np.std(ts_mat[count, :])
        else:
            var_mat[count] = np.var(ts_mat[count, :])
    np.savetxt('%s.txt' %outfile, var_mat)

def princomp(A,numpc=10):
    """
    Run principal components analysis on an input matrix
    """
    from numpy import mean,cov,cumsum,dot,linalg,size,flipud
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-np.mean(A.T,axis=1).T) # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M))
    p = size(coeff,axis=1)
    idx = np.argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues
    if numpc < p or numpc >= 0:
        coeff = coeff[:,range(numpc)] # cutting some PCs
    score = dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent

def whole_brain_degree(fmri_file,out_file=None,nuisance_file=False,mask_file=False,edge_threshold=0):
    """Compute voxelwise whole brain degree

    Parameters
    ----------
    fmri_file: string
      The path to the fMRI 4D .nii file
    out_file: string, optional
      The path/name of the output .nii file. If not specified, output is 'whole_brain_degree.nii.gz'
    nuisance_file : string, optional
      The path to the nuisance parameters text file. If True regress out nuisance parameters first.
    mask_file: string, optional
      The path to a mask .nii file. If True only calculate whole brain degree amongst voxels in mask.
    edge_threshold: num, optional
      The r-value cutoff for determining whole brain degree; if specified, binary count of edges; otherwise, non-binary sum

    Returns
    -------

    """
    # Determine subset of voxels for which every voxel has nonzero values and optionally is in mask
    input = nib.load(fmri_file)
    input_d = input.get_data()
    data_sum = np.sum(input_d,axis=3)
    data_sum_flat = data_sum.flatten()
    data_nzs = np.nonzero(data_sum_flat)

    if mask_file:
        mask = nib.load(mask_file)
        mask_d = mask.get_data()
        mask_d_flat = mask_d.flatten()
        mask_nzs = np.nonzero(mask_d_flat)
        keep_vox = list(set(data_nzs[0].tolist()) & set(mask_nzs[0].tolist()))
    else:
        keep_vox = data_nzs[0].tolist()
    keep_vox.sort()
    keep_vox_array = np.array(keep_vox)
    n_vox = len(keep_vox)

    dims = input_d.shape
    input_d_flat = np.reshape(input_d, (dims[0]*dims[1]*dims[2],dims[3]))
    input_d_flat_trim = input_d_flat[keep_vox,:]

    del(input,input_d,input_d_flat,mask,mask_d,mask_d_flat)

    # calculate voxelwise correlation matrix
    r_mat = np.zeros((n_vox,n_vox))
    if nuisance_file:
        nuis_reg = np.array(core.file_reader(nuisance_file)).T
        input_d_flat_trim_res = np.zeros((input_d_flat_trim.shape))
        for i in range(len(input_d_flat_trim)):
            ts = input_d_flat_trim[i,:]
            reg = np.linalg.lstsq(nuis_reg.T,ts.T) # regress out nuisance parameters
            beta = reg[0]
            input_d_flat_trim_res[i,:] = np.squeeze(ts.T - nuis_reg.T.dot(beta)) # store residuals
        del(input_d_flat_trim)
        r_mat = np.corrcoef(input_d_flat_trim_res)
    else:
        r_mat = np.corrcoef(input_d_flat_trim)

    r_mat = r_mat[0:len(keep_vox),0:len(keep_vox)]
    r_mat = np.nan_to_num(r_mat)

    # calculate whole brain degree
    if edge_threshold > 0:
        whole_brain_degree_vals = np.sum((r_mat > edge_threshold),axis=0) # binary count
    else:
        whole_brain_degree_vals = np.sum(r_mat,axis=0) # non-binary (weighted) sum

    # get x,y,z values for each index
    l = np.unravel_index(keep_vox_array,(dims[0],dims[1],dims[2]))

    wb_deg_img = np.zeros((dims[0],dims[1],dims[2]))
    wb_deg_img[l[0].tolist(),l[1].tolist(),l[2].tolist()] = whole_brain_degree_vals.tolist()
    mni_img = nib.load('/data/mridata/jbrown/brains/MNI152_T1_4mm_brain.nii.gz')
    img = nib.Nifti1Image(wb_deg_img, mni_img.get_affine())
    if out_file:
        img.to_filename(out_file)
    else:
        img.to_filename("whole_brain_degree.nii.gz")

def mask_funcconnec_matrix_sliding(nifti_file,masks_files,outfile=None,masks_threshes = [],
                           multi_labels=[],zero_diag=True,ts_outfile=None,covariate_ts_file=None,
                           window_length=30):
    """
    Calculates correlation matrix for a set of mask mean timeseries'
    masks_files: list of mask filenames with full path, can either be one mask
                 per file (in which case multi_labels should be []) or one file
                 with multiple numerical labels (multi_labels = [num1, num2, ...])
    masks_threshes: list of numerical values to use as lower threshold for separate
                    mask files
    covariate_ts_file: text file with timeseries for nuisance covariates to partial out
    window_length: the number of volumes to include in a sliding window correlation
    output options:
    1) correlation matrix
    """
    if multi_labels:
        masks_coords = core.get_mask_labels(masks_files[0], labels=multi_labels)
    else:
        if masks_threshes:
            masks_coords = []
            for count, mask in enumerate(masks_files):
                masks_coords.append(core.get_nonzero_coords(mask, masks_threshes(count)))
        else:
            masks_coords = [core.get_nonzero_coords(mask) for mask in masks_files]
    n_regions = len(masks_coords)
    input = nib.load(nifti_file)
    input_d = input.get_data()
    if len(input.shape) > 3:
        ts_length = input.shape[3]
    else:
        ts_length = 1
    masks_mean_ts_array = np.zeros((len(masks_coords), ts_length))
    for count, mask_coords in enumerate(masks_coords):
        mask_array = [input_d[mask_coord[0], mask_coord[1], mask_coord[2], :] for mask_coord in mask_coords]
        masks_mean_ts_array[count, :] = np.mean(mask_array, axis=0)
    if covariate_ts_file:
        nuis_reg = np.array(core.file_reader(covariate_ts_file))
        masks_mean_ts_array_resid = np.zeros((n_regions,ts_length))
        for i in range(n_regions):
            ts1 = np.atleast_2d(masks_mean_ts_array[i,:])
            reg = np.linalg.lstsq(nuis_reg,ts1.T)
            beta = reg[0]
            ts1_resid = np.squeeze(ts1.T - nuis_reg.dot(beta))
            masks_mean_ts_array_resid[i,:] = ts1_resid
    n_windows = len(range(ts_length-window_length))
    mats = np.zeros((n_regions,n_regions,n_windows))
    for k in range(ts_length-window_length):
        mat = np.zeros((n_regions,n_regions))
        ts_start = k
        ts_stop = k + window_length
        if covariate_ts_file:
            mat = np.corrcoef(masks_mean_ts_array_resid[:,ts_start:ts_stop])
        else:
            mat = np.corrcoef(masks_mean_ts_array[:,ts_start:ts_stop])
        if zero_diag:
            mat = mat * abs(1-np.eye(mat.shape[0])) # zero matrix diagonal
        mats[:,:,k] = mat
    if outfile:
        mats_2d = np.reshape(mats,[n_regions,n_regions*n_windows],'F').T # stack matrices vertically
        np.savetxt('%s.txt'%outfile, mats_2d)
    if ts_outfile:
        np.savetxt('%s.txt'%ts_outfile, masks_mean_ts_array)
    return mats, masks_mean_ts_array