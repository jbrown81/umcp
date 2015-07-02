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

import os
import sys
import struct
import numpy as np
import nibabel as nib
import core

def get_floats(track_file):
    """Read in all tracks from a .trk (TrackVis) file and store them in a list"""
    track_list = []
    header_dict = get_header(track_file)
    size = os.path.getsize(track_file)
    n_s = header_dict["n_scalars"]
    n_p = header_dict["n_properties"]
    f = open(track_file, 'rb') # added 'rb' for Windows reading
    contents = f.read(size)
    current = 1000
    end = current + 4
    while end < size:
        length = struct.unpack('i', contents[current:end])[0]
        current = end
        distance = length*(12+(4*n_s)) # modify for scalars here
        #distance=length*12
        end = current + distance
        if end > size:
            break
        floats = []
        float_range = range(current,end,4)
        for float_start in float_range:
            float_end = float_start + 4
            floats.append(struct.unpack('f',contents[float_start:float_end])[0])
        
        if n_p: # if track has at least one property; currently not storing properties
            properties_start = float_end
            property_start = properties_start
            #track_properties = []
            for p in range(n_p):
                property_end = property_start + 4
                #track_properties.append(struct.unpack('f',contents[property_start:property_end])[0])
                property_start = property_end + 4
        
        floats = zip(*[iter(floats)] * 3)
        current = end + (4 * n_p)
        end = current + 4
        if len(floats) > 0:
            track_list.append(floats)
        l = len(track_list)
    f.close()
    return track_list

def get_header(track_file):
    """Read in header values from a .trk (TrackVis) file and store them in a dictionary"""
    header_dict={}
    f=open(track_file, 'rb') # added 'rb' for Windows reading
    contents = f.read()
    dims=(struct.unpack('h',contents[6:8])[0],struct.unpack('h',contents[8:10])[0],struct.unpack('h',contents[10:12])[0])   
    header_dict["dims"]=dims
    vox_size=(struct.unpack('f',contents[12:16])[0],struct.unpack('f',contents[16:20])[0],struct.unpack('f',contents[20:24])[0])
    header_dict["vox_size"]=vox_size
    origin=(struct.unpack('f',contents[24:28])[0],struct.unpack('f',contents[28:32])[0],struct.unpack('f',contents[32:36])[0])
    header_dict["origin"]=origin
    n_scalars=(struct.unpack('h',contents[36:38]))[0]
    header_dict["n_scalars"]=n_scalars
    n_properties=(struct.unpack('h',contents[238:240]))[0]
    header_dict["n_properties"]=n_properties
    vox_order=(struct.unpack('c',contents[948:949])[0],struct.unpack('c',contents[949:950])[0],struct.unpack('c',contents[950:951])[0])    
    header_dict["vox_order"]=vox_order
    paddings=(struct.unpack('c',contents[952:953])[0],struct.unpack('c',contents[953:954])[0],struct.unpack('c',contents[954:955])[0])
    header_dict["paddings"]=paddings
    img_orient_patient=(struct.unpack('f',contents[956:960])[0],struct.unpack('f',contents[960:964])[0],struct.unpack('f',contents[964:968])[0],\
          struct.unpack('f',contents[968:972])[0],struct.unpack('f',contents[972:976])[0],struct.unpack('f',contents[976:980])[0])
    header_dict["img_orient_patient"]=img_orient_patient
    inverts=(struct.unpack('B',contents[982:983])[0],struct.unpack('B',contents[983:984])[0],struct.unpack('B',contents[984:985])[0])
    header_dict["inverts"]=inverts
    swaps=(struct.unpack('B',contents[985:986])[0],struct.unpack('B',contents[986:987])[0],struct.unpack('B',contents[987:988])[0])
    header_dict["swaps"]=swaps
    num_fibers=(struct.unpack('i',contents[988:992])[0])
    header_dict["num_fibers"]=num_fibers
    f.close()
    return header_dict

def mm_to_vox_convert(tracks,header,dsi_studio=False):
    """Convert track coordinates from mm dimensions to voxel dimensions"""
    xsize,ysize,zsize=np.array(header["vox_size"])
    if dsi_studio:
        # hack for my 96x96x48 LPS oriented trk files created by dsi_studio
        tracks_new = [[(int(x//xsize),int((240-y)//ysize),int(z//zsize)) for x,y,z in track] for track in tracks]
    else:
        tracks_new = [[(int(x//xsize),int(y//ysize),int(z//zsize)) for x,y,z in track] for track in tracks]
    return tracks_new

def add_missing_vox(tracks):
    """Add voxels between track points that are separated by more than 1 voxel in
    x, y, or z directions"""
    tracks_filled = []
    for track in tracks:   
        track_vox_set = set(track)
        new_track = []
        for p in range(len(track) - 1):
            new_track.append(track[p])
            a = np.array(track[p])
            b = np.array(track[p + 1]) 
            dif = b - a
            ranges = []
            if any(abs(dif) >= 2):
                for count,val in enumerate(dif):
                    if val <= -2:
                        ranges.append(range(a[count],a[count] + val,-1))
                    elif val >= 2:
                        ranges.append(range(a[count],a[count] + val))
                    elif val < 0:
                        ranges.append(range(a[count],a[count] - 1,-1))
                    else:
                        ranges.append(range(a[count],a[count] + 1))
                missing_vox_set = set([(x,y,z) for x in ranges[0] for y in ranges[1] for z in ranges[2]])
                new_missing_vox = list(missing_vox_set - track_vox_set)
                new_track.extend(new_missing_vox)
        new_track.append(track[-1])
        tracks_filled.append(new_track)
    return tracks_filled
    
def mask_tracks(tracks,header,masks,nonzero_thresh=0,through=1,write_nii=0,outprefix="mask",tracks_mm=0,length_thresh=0):
    """
    Creates density files for all tracks passing through a set of masks
    """
    # Each volume in vox_tracks_img is the density volume for a single mask
    # Leave 'through' argument as 0 to count number of tracks that originate/terminate 
    # within a mask, set through to 1 to count number of tracks that intersect a mask
    xdim,ydim,zdim=header["dims"]
    mm_dims=np.array(header["vox_size"])*np.array(header["dims"])
    masks_coords_list=[]
    if write_nii == 1:
        vox_tracks_img=np.zeros((xdim,ydim,zdim,len(masks)))
    tracknums=[[] for x in range(len(masks))]
    for mask in masks:
        masks_coords_list.append(set(core.get_nonzero_coords(mask,nonzero_thresh)))
    for tracknum,track in enumerate(tracks):
        if through == 0:
            track_start_set=set([track[0]])
            track_end_set=set([track[-1]])
            for count,mask_coords_set in enumerate(masks_coords_list):
                if track_start_set & mask_coords_set or track_end_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            tracknums[count].append(tracknum)
                        if write_nii==1:
                            for x,y,z in track:
                                if all(np.array([x,y,z])<mm_dims):
                                    vox_tracks_img[x,y,z,count] += 1
                    else:
                        tracknums[count].append(tracknum)
                        if write_nii==1:
                            for x,y,z in track:
                                if all(np.array([x,y,z])<mm_dims):
                                    vox_tracks_img[x,y,z,count] += 1
        elif through == 1:  
            track_set=set(track)
            # track_set = track
            for count,mask_coords_set in enumerate(masks_coords_list):
                if track_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            tracknums[count].append(tracknum)
                        if write_nii==1:
                            for x,y,z in track:
                                if all(np.array([x,y,z])<mm_dims):
                                    vox_tracks_img[x,y,z,count] += 1
                    else:
                        tracknums[count].append(tracknum)
                        if write_nii==1:
                            for x,y,z in track:
                                if all(np.array([x,y,z])<mm_dims):
                                    vox_tracks_img[x,y,z,count] += 1
    mask_density = [len(hits) for hits in tracknums]
    if write_nii == 0:
        np.savetxt('%s_density.txt'%outprefix,mask_density)
        return tracknums
    else:
        outnifti = nib.Nifti1Image(vox_tracks_img, np.eye(4))
        outnifti.to_filename('%s_density.nii'%outprefix)
        return tracknums

def mask_connectivity_matrix(tracks,header,masks,outfile,nonzero_thresh=0,through=0,tracks_mm=0,length_thresh=0,
                             mask_matrix_file=None,write_tracks=False,write_tracks_filename=None,track_file=None):
    """
    Calculate the (symmetric) connectivity matrix for a set of tracks (from diffusion toolkit .trk file) and a
    set of masks
    """
    # Leave third argument as 0 to count number of tracks that originate/terminate at
    # either end of a pair of masks, set through to 1 to count number of tracks that
    # intersect both of the masks.
    connect_mat=np.zeros((len(masks),len(masks)))
    masks_coords_list=[]
    tracknums=[[] for x in range(len(masks)*len(masks))]
    for mask in masks:
        masks_coords_list.append(set(core.get_nonzero_coords(mask,nonzero_thresh)))
    if mask_matrix_file:
        mask_matrix = core.file_reader(mask_matrix_file)
        mask_matrix_array = np.array(mask_matrix)
    for tracknum,track in enumerate(tracks):
        if through == 0:
            cur_start=[]
            cur_end=[]
            track_start_set=set([track[0]])
            track_end_set=set([track[-1]])      
            for count,mask_coords_set in enumerate(masks_coords_list):
                if track_start_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            cur_start.append(count)
                    else:
                        cur_start.append(count)
                elif track_end_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            cur_end.append(count)
                    else:
                        cur_end.append(count)
            for x in cur_start:
                for y in cur_end:
                    # allow for fiber to start/end in multiple (overlapping) masks
                    if mask_matrix_file:
                        if mask_matrix_array[x,y]:
                            connect_mat[x,y] += 1
                            tracknums[(x*len(masks))+y].append(tracknum)
                    else:
                       connect_mat[x,y] += 1
                       tracknums[(x*len(masks))+y].append(tracknum)
        elif through == 1:
            cur=[]
            track_set=set(track)
            for count,mask_coords_set in enumerate(masks_coords_list):
                if track_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            cur.append(count)
                    else:
                        cur.append(count)
            for x,y in list(core.combinations(cur,2)):
                if mask_matrix_file:
                    if mask_matrix_array[x,y]:
                        connect_mat[x,y] += 1
                        tracknums[(x*len(masks))+y].append(tracknum)
                else:
                    connect_mat[x,y] += 1
                    tracknums[(x*len(masks))+y].append(tracknum)
    connect_mat_sym = core.symmetrize_mat_sum(connect_mat)
    tracknums_sym = core.symmetrize_tracknum_list(tracknums)
    np.savetxt('%s_connectmat.txt'%outfile,connect_mat_sym)
    if write_tracks:
        tracknum_list = list(set([item for sublist in tracknums for item in sublist]))
        tracknum_list_ordered = sorted(tracknum_list)
        track_list = [tracks_mm[n] for n in tracknum_list_ordered]
        make_floats(track_list,write_tracks_filename,track_file)
    return connect_mat_sym,tracknums_sym 

def tracklength(track):
    track_len = 0
    for i in range(len(track)):
        a = track[i]
        if i < len(track) - 1: # for length calcs
            b = track[i + 1]
            ab = a - b
            track_len = track_len + np.sqrt(np.dot(ab,ab))
    return track_len

def trackcurve(track):
    track_curve = 0
    for i in range(len(track)):
        if i < len(track)-2: # for angle calcs
            a = track[i]
            b = track[i + 1]
            ab = a - b
            c = track[i + 2]
            bc = c - b
            track_curve = track_curve + \
               np.arccos(np.dot(ab,bc)/(np.sqrt(np.dot(ab,ab))*np.sqrt(np.dot(bc,bc))))
    track_curve = track_curve * 180/np.pi
    return track_curve

def track_stats(tracknums,tracks_mm,header,vox_volume,vox_dims,tracks_vox=0,statimage=0,statimage_data=[]):
    """Given a list of track numbers and the tracks object with mm coordinates,
    calculate statistics for the tracks: total volume, avg track length,
    avg_track_curvature, and (optionally, requires input image) avg value from a
    statistical image such as FA, MD"""
    # NOTE: track curvature, length, volume calculated from tracks_mm
    # stats from statsimage calculated from tracks_vox
    track_vols = []
    track_lens = []
    track_curves = []
    if statimage:
        track_imagevals_list = []
    if len(statimage_data) > 0: # if statimage_data is pre-loaded
        pass
    else:
        input = nib.load(statimage)
        statimage_data = input.get_data()
    for tracknum in tracknums:
        track = np.array(tracks_mm[tracknum])
        trackvoxcount = len(track)
        trackvoxcount_adjusted = trackvoxcount
        track_len = tracklength(track)
        track_curve = trackcurve(track)
        track_vol = len(track)*vox_volume
        track_imageval_cur = 0
        for i in xrange(len(track)):
            if statimage:
                x2,y2,z2=tracks_vox[tracknum][i] # coords from tracks_vox
                if x2>(vox_dims[0]-1) or y2>(vox_dims[1]-1) or z2>(vox_dims[2]-1):
                    # exclude tracks who go outside the dimensions of the statimage
                    trackvoxcount_adjusted = trackvoxcount_adjusted - 1
                    pass
                else:
                    track_imageval_cur = track_imageval_cur + statimage_data[x2,y2,z2]
        track_vols.append(track_vol)
        track_lens.append(track_len)
        track_curves.append(track_curve * 180/np.pi)
        if statimage:
            track_imagevals_list.append([len(track),track_imageval_cur])
    total_vol = sum(track_vols)
    track_curves = [z for z in track_curves if np.isnan(z) != 1] # if angle is nan
    if len(track_lens)>0:
        avg_distance = sum(track_lens)/len(track_lens)
        # avg_distance_std = np.std(track_lens)
    else:
        avg_distance = 0
    if len(track_curves) > 0:   
        avg_curve = sum(track_curves)/len(track_curves)
    else:    
        avg_curve = 0
    if statimage:
        if len(track_imagevals_list)>0:
            vox_counts,val_sums = zip(*track_imagevals_list)
            avg_imageval = sum(val_sums) / sum(vox_counts) # weighted average
        else:
            avg_imageval=0
        return total_vol,avg_distance,avg_curve,avg_imageval # avg_distance_std
    else:
        return total_vol,avg_distance,avg_curve # avg_distance_std

def track_stats_list(tracknums_list,tracks_mm,header,outprefix,tracks_vox=0,statimage=0):
    """Calculate matrices for total track bundle volume, avg track length,
    avg track curvature, and optionally an average track statistic from a statistical
    image like FA or MD
    Requires tracknums_list output by mask_connectivity_matrix"""
    xdim = len(tracknums_list)
    volumelist = np.zeros((xdim))
    lengthlist = np.zeros((xdim))
    curvelist = np.zeros((xdim))
    statlist = np.zeros((xdim))
    xsize,ysize,zsize = header["vox_size"]
    vox_volume = xsize * ysize * zsize
    vox_dims = header["dims"]
    if statimage:
        input = nib.load(statimage)
        statimage_data = input.get_data()
    for i in range(xdim):
        if statimage:
            volumelist[i],lengthlist[i],curvelist[i],statlist[i] = track_stats(tracknums_list[i],tracks_mm,header,vox_volume,vox_dims,tracks_vox,statimage,statimage_data)
        else:
            volumelist[i],lengthlist[i],curvelist[i] = track_stats(tracknums_list[i],tracks_mm,header,vox_volume,vox_dims)
    if statimage:
        np.savetxt('%s_volumelist.txt'%outprefix,volumelist)
        np.savetxt('%s_lengthlist.txt'%outprefix,lengthlist)
        np.savetxt('%s_curvelist.txt'%outprefix,curvelist)
        np.savetxt('%s_statlist.txt'%outprefix,statlist)
        return volumelist,lengthlist,curvelist,statlist
    else:
        np.savetxt('%s_volumelist.txt'%outprefix,volumelist)
        np.savetxt('%s_lengthlist.txt'%outprefix,lengthlist)
        np.savetxt('%s_curvelist.txt'%outprefix,curvelist)
        return volumelist,lengthlist,curvelist
    
def track_stats_group(tracknums_list,tracks_mm,header,outprefix,tracks_vox=0,statimage=0):
    """Calculate matrices for total track bundle volume, avg track length,
    avg track curvature, and optionally an average track statistic from a statistical
    image like FA or MD
    Requires tracknums_list output by mask_connectivity_matrix"""
    # Run the trackstats function for a list of track number lists output by
    # mask_connectivity_matrix"""
    xdim = np.sqrt(len(tracknums_list)).astype('int')
    ydim = xdim
    volumemat = np.zeros((xdim,ydim))
    lengthmat = np.zeros((xdim,ydim))
    curvemat = np.zeros((xdim,ydim))
    statmat = np.zeros((xdim,ydim))
    xsize,ysize,zsize = header["vox_size"]
    vox_volume = xsize * ysize * zsize
    vox_dims = header["dims"]
    if statimage:
        input = nib.load(statimage)
        statimage_data = input.get_data()
    for i in xrange(xdim-1):
        for j in xrange(i+1,ydim):
            index = (i*xdim)+j
            if statimage:
                volumemat[i,j],lengthmat[i,j],curvemat[i,j],statmat[i,j]=\
                track_stats(tracknums_list[index],\
                tracks_mm,\
                header,\
                vox_volume, vox_dims,\
                tracks_vox,\
                statimage,statimage_data)
            else:
                volumemat[i,j],lengthmat[i,j],curvemat[i,j]=\
                track_stats(tracknums_list[index],\
                tracks_mm,\
                header,vox_volume,vox_dims)
    volumemat = core.symmetrize_mat(volumemat,'top')
    lengthmat = core.symmetrize_mat(lengthmat,'top')
    curvemat = core.symmetrize_mat(curvemat,'top')
    if statimage:
        statmat=core.symmetrize_mat(statmat,'top')
        np.savetxt('%s_volumemat.txt'%outprefix,volumemat)
        np.savetxt('%s_lengthmat.txt'%outprefix,lengthmat)
        np.savetxt('%s_curvemat.txt'%outprefix,curvemat)
        np.savetxt('%s_statmat.txt'%outprefix,statmat)
        return volumemat,lengthmat,curvemat,statmat
    else:
        np.savetxt('%s_volumemat.txt'%outprefix,volumemat)
        np.savetxt('%s_lengthmat.txt'%outprefix,lengthmat)
        np.savetxt('%s_curvemat.txt'%outprefix,curvemat)
        return volumemat,lengthmat,curvemat

def get_tracks_dsi_studio(tracks_file,xsize=2.5,ysize=2.5,zsize=2.5):
    """
    Read tracks from DSI studio tracks .txt file
    xsize, ysize, zsize specify voxel size
    """
    mm_convert = False
    tracks = core.file_reader(tracks_file)
    tracks_new = []
    for track in tracks:
        track_new = []
        track_len = len(track) / 3
        for count in range(track_len):
            start = (count * 3)
            if mm_convert:
                track_new.append((int(track[start]//xsize), int(track[start+1]//ysize), int(track[start+2]//zsize)))
            else:
                track_new.append((int(96-track[start]), int(96-track[start+1]), int(track[start+2])))
        tracks_new.append(track_new)
    return tracks_new

def mask_connectivity_matrix_dsi(tracks,masks,outfile,nonzero_thresh=0,through=0,tracks_mm=0,length_thresh=0,header=None):
    """Calculate the (symmetric) connectivity matrix for a set of tracks from a DSI studio .txt file and a
    set of masks"""
    # Leave third argument as 0 to count number of tracks that originate/terminate at
    # either end of a pair of masks, set through to 1 to count number of tracks that
    # intersect both of the masks.
    connect_mat=np.zeros((len(masks),len(masks)))
    masks_coords_list=[]
    tracknums=[[] for x in range(len(masks)*len(masks))]
    for mask in masks:
        masks_coords_list.append(set(core.get_nonzero_coords(mask,nonzero_thresh)))
    for tracknum,track in enumerate(tracks):
        if through == 0:
            cur_start=[]
            cur_end=[]
            track_start_set=set([track[0]])
            track_end_set=set([track[-1]])      
            for count,mask_coords_set in enumerate(masks_coords_list):
                if track_start_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            cur_start.append(count)
                    else:
                        cur_start.append(count)
                elif track_end_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            cur_end.append(count)
                    else:
                        cur_end.append(count)
                for x in cur_start:
                    for y in cur_end:
                        # allow for fiber to start/end in multiple (overlapping) masks
                        connect_mat[x,y] += 1
                        tracknums[(x*len(masks))+y].append(tracknum)
        elif through == 1:
            cur=[]
            track_set=set(track)
            for count,mask_coords_set in enumerate(masks_coords_list):
                if track_set & mask_coords_set:
                    if length_thresh:
                        track_len = tracklength(np.array(tracks_mm[tracknum]))
                        if track_len > length_thresh:
                            cur.append(count)
                    else:
                        cur.append(count)
            for x,y in list(core.combinations(cur,2)):
                connect_mat[x,y] += 1
                tracknums[(x*len(masks))+y].append(tracknum)
    connect_mat_sym = core.symmetrize_mat_sum(connect_mat)
    tracknums_sym = core.symmetrize_tracknum_list(tracknums)
    np.savetxt('%s_connectmat.txt'%outfile,connect_mat_sym)
    return connect_mat_sym,tracknums_sym

def make_floats(track_list,output_filename,input_trackfile):
    """
    Take a track list and generate a .trk (TrackVis) file
    """
    # can copy header from input file if input file exists
    # should only need to change num_fibers
    # no real point in generating full file from scratch at this point
    f = open(input_trackfile, 'rb') # added 'rb' for Windows reading
    contents = f.read()
    f.close()
    header = contents[0:1000]
    
    outfile = open(output_filename,'wb')
    outfile.write(header[0:988])
    
    num_fibers = len(track_list)
    num_fibers_packed = struct.pack('i',num_fibers)
    outfile.write(num_fibers_packed)
    outfile.write(header[992:1000])
    
    for track in track_list:
        track_n_points = struct.pack('i',len(track))
        outfile.write(track_n_points)
        for point in track:
            for coord in point:
                cur_float = struct.pack('f',coord)
                outfile.write(cur_float) # do i need to specify length or just append?
    outfile.close()