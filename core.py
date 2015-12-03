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
#    Copyright 2015 Jesse Brown

import os
import sys
import numpy as np
import nibabel as nib
import re
from commands import getoutput

def get_nonzero_coords(nifti_file,thresh=0,value=0):
    """
    Given a .nii file name, return a list of all the coordinates with non-zero (or above threshold) values
    """
    input = nib.load(nifti_file)
    input_d = input.get_data()
    if len(input.shape) > 3:
        ts_length = input.shape[3]
    else:
        ts_length = 1
    nonzero_coords=[]
    if ts_length > 1:
        nonzero_sums = np.nonzero(input_d.sum(axis=3))
        nonzero_coords = zip(*nonzero_sums)
    else:
        if value:
            nonzeros = np.nonzero(input_d==value)
            nonzero_coords = zip(*nonzeros)
        else:
            nonzeros = np.nonzero(input_d>thresh)
            nonzero_coords = zip(*nonzeros)
    return nonzero_coords

def match_nifti_header(data, input_file):
    """
    Takes NiftiImage object and input file, updates NiftiImage header to match input file header
    """
    img_to_match = nib.load(input_file)
    header_to_match = img_to_match.get_header()
    img_new = nib.Nifti1Image(data, header_to_match.get_best_affine(), header = img_to_match.get_header())
    return img_new

def symmetrize_mat(inarray,type):
    """
    Given a 2D numpy array, create a symmetric equivalent by mirroring and transposing
    either the upper or lower triangle
    """
    x,y = inarray.shape
    outarray = np.zeros((x,y))
    if type=='top':
            for i in range(x-1):
                    for j in range(i+1,y):
                            outarray[i,j]=inarray[i,j]
                            outarray[j,i]=inarray[i,j]
    elif type=='bottom':
            for i in range(x-1):
                    for j in range(i+1,y):
                            outarray[i,j]=inarray[j,i]
                            outarray[j,i]=inarray[j,i]
    return outarray

def symmetrize_mat_sum(inarray):
    x,y=inarray.shape
    outarray=np.zeros((x,y))
    for i in range(x-1):
            for j in range(i+1,y):
                    outarray[i,j]=inarray[i,j]+inarray[j,i]
                    outarray[j,i]=inarray[i,j]+inarray[j,i]
    return outarray

def symmetrize_tracknum_list(inkeeptracks):
    masknum = np.sqrt(len(inkeeptracks)).astype('int')
    x,y = masknum,masknum
    outkeeptracks = [[] for z in range(len(inkeeptracks))]
    for i in range(x-1):
            for j in range(i+1,y):
                    outkeeptracks[(i*masknum)+j]=inkeeptracks[(i*masknum)+j] + inkeeptracks[(j*masknum)+i]
                    outkeeptracks[(j*masknum)+i]=inkeeptracks[(i*masknum)+j] + inkeeptracks[(j*masknum)+i]
    return outkeeptracks

def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
            return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
            for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                            break
            else:
                    return
            indices[i] += 1
            for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)

def file_reader(infile,text=False):
    """
    Read in a .txt file containing strings or numbers
    """
    fin = open(infile,'rU')
    values = []
    if text:
        for line in fin:
            pos = line.rstrip()
            values.append(pos)
    else:
        for line in fin:
            pos = line.rstrip().split()
            values.append(map(float, pos))
    fin.close()
    return values

def prob_tensor(dyadfile, ffile, outname):
    """
    Given dyads<i>.nii.gz and mean_f<i>samples.nii.gz files from FSL bedpostx,
    generate a simulated tensor file where first eigenvector is vector from dyad
    and first eigenvalue is value from mean_f sample; tensor file can be input
    into Diffusion Toolkit for deterministic tractography
    This method was used in http://www.pnas.org/content/108/51/20760.long
    """
    dyad = nib.load(dyadfile)
    dyaddata = dyad.get_data()
    s1, s2, s3, s4 = dyaddata.shape
    f = nib.load(ffile)
    fdata = f.get_data()

    tensor_img = np.zeros((s1,s2,s3,6)) # output simulated tensor image

    for dim1 in range(s1):
        for dim2 in range(s2):
            for dim3 in range(s3):
                v1=dyaddata[:,dim3,dim2,dim1] # dyad vector, treated as first eigenvector
                e1=fdata[dim3,dim2,dim1] # dyad mean value, treated as first eigenvalue
                e2=e1/2 # random second eigenvalue, less than e1
                e3=e2 # random third eigenvalue, less than e1
                D=np.array([e1,0,0,0,e2,0,0,0,e3]).reshape(3,3)

                a,b,c=v1
                x=np.random.rand(1)
                y=np.random.rand(1)
                z=(-(a*x)-(b*y))/c # arbitrary solution to dot([a,b,c],[x,y,z]) = 0
                v2=np.array([x,y,z]).T # random second eigenvector, perpendicular to v1
                v2=v2[0]
                v2=v2/np.linalg.norm(v2)
                v3=np.cross(v1,v2) # random third eigenvector, perpendicular to v1
                V=np.array([v1,v2,v3]).T

                tensor=np.matrix(V)*np.matrix(D)*np.matrix(V).T # tensor formula: S = V*D*V.T
                tensor_vals=[tensor[0,0], tensor[0,1], tensor[1,1], tensor[0,2], tensor[1,2], tensor[2,2]]
                tensor_img[:,dim3,dim2,dim1]=tensor_vals

    outnifti = nib.Nifti1Image(tensor_img, dyad.get_header().get_best_affine())
    outnifti.to_filename(outname)

def partialcorr(a,b,c):
    """
    Calculate partial correlation between 1D arrays a and b, controlling for c
    """
    import scipy.linalg
    import scipy.stats
    x1, res1, rank1, s1 = scipy.linalg.lstsq(c,a)
    x2, res2, rank2, s2 = scipy.linalg.lstsq(c,b)
    r1 = a-(c*x1.T).sum(1)
    r2 = b-(c*x2.T).sum(1)
    coef, p = scipy.stats.pearsonr(r1,r2)
    return coef, p

def partialcorr_matrix(ts_array):
    """
    Take a timeseries array and calculate the partial correlation matrix
    ts_array must be regions x timepoints
    Adapted from: http://www.fmrib.ox.ac.uk/analysis/netsim/
    """
    import scipy.linalg
    num_nodes = ts_array.shape[0]
    ic = -scipy.linalg.inv(np.cov(ts_array))
    d = np.sqrt(abs(np.diag(ic)))

    x = np.tile(d,[num_nodes,1])
    y = np.tile(d,[num_nodes,1]).T

    r = ic / x / y
    r = r + np.identity(num_nodes)
    return r

def get_mask_labels(nifti_file, labels=[0]):
    """
    Given a nifti file with integer labels for different masks
    Return a list which contains for each mask, a list of (x,y,z) coordinates
    """
    nonzero_coords_all = []
    for l in labels:
            nonzero_coords_all.append(get_nonzero_coords(nifti_file,value=l))
    return nonzero_coords_all

def my_scoreatpercentile(in_mat, threshold_pct):
    """
    Given a matrix and an integer percentage value (between 0-100),
    threshold the matrix to keep only the threshold_pct largest values,
    return the cutoff value
    """
    a = in_mat.ravel()
    a_sort = a[a.argsort()[::-1]]
    dims = in_mat.shape
    dim_x = dims[0] # only need first dimension because matrices are always square
    if len(dims) > 1:
        thresh_value_pre = (threshold_pct/100.)*(dim_x*dim_x) # index of cutoff value
    else:
        thresh_value_pre = (threshold_pct/100.) * dim_x
    thresh_value = (round(thresh_value_pre / 2) * 2)
    if len(dims) > 1:
        if thresh_value >= dim_x*dim_x:
            cutoff = min(a_sort)
        else:
            cutoff = a_sort[thresh_value]
    else:
        if thresh_value >= dim_x:
            cutoff = min(a_sort)
        else:
            cutoff = a_sort[thresh_value]
    return cutoff

def euclidean_distance(coords):
    """
    Given a iX3 numpy array of (x,y,z) coordinates
    Calculate the euclidean distance between all pairs of coordinates
    Return a distance matrix
    """
    numcoords = coords.shape[0]
    eucdistmat = np.zeros((numcoords, numcoords))

    for i in range(numcoords):
        for j in range(numcoords):
            x = coords[i, :]
            y = coords[j, :]
            eucdistmat[i,j] = np.sqrt(((y[0]-x[0])**2) + ((y[1]-x[1])**2) + ((y[2]-x[2])**2))
    return eucdistmat

def global_efficiency(G, regional=False, weight=True):
    e_regs = []
    avg = 0.0
    n = len(G)
    for node in G:
        if weight:
            path_length = nx.shortest_path_length(G, node, weight='weight')
        else:
            path_length = nx.shortest_path_length(G, node)
        cur_pathweight = sum(1.0/v for v in path_length.values() if v !=0)
        e_regs.append(cur_pathweight/(n-1))
        avg += cur_pathweight
    avg *= 1.0/(n*(n-1))
    e_glob = avg
    if regional:
        return e_glob, e_regs
    else:
        return e_glob

def participation_coefficient(G, weighted_edges=False):
    """"Compute participation coefficient for nodes.
    
    Parameters
    ----------
    G: graph
      A networkx graph
    weighted_edges : bool, optional
      If True use edge weights
    
    Returns
    -------
    node : dictionary
      Dictionary of nodes with participation coefficient as the value
    
    Notes
    -----
    The participation coefficient is calculated with respect to a community
    affiliation vector. This function uses the community affiliations as determined
    by the Louvain modularity algorithm (http://perso.crans.org/aynaud/communities/).
    """
    partition = community.best_partition(G)
    partition_list = []
    for count in range(len(partition)):
        partition_list.append(partition[count])

    n = G.number_of_nodes()
    Ko = []
    for node in range(n):
        node_str = np.sum([G[node][x]['weight'] for x in G[node].keys()])
        Ko.append(node_str)
    Ko = np.array(Ko)
    G_mat_weighted = np.array(nx.to_numpy_matrix(G))
    G_mat = (G_mat_weighted != 0) * 1
    D = np.diag(partition_list)
    Gc = np.dot(G_mat, D)
    Kc2 = np.zeros(n)
    for i in range(np.max(partition_list) + 1):
        Kc2 = Kc2 + (np.sum(G_mat_weighted * (Gc == i),1) ** 2)

    P = np.ones(n) - (Kc2/(Ko **2))
    return P

def maxprob(coord):
	"""
	Given a x,y,z coordinate in MNI152 space, use FSL atlasquery to find maximum
	probability region
	"""
	# this can be extended for any atlas that atlasquery uses, including Talairach Daemon
	print coord
	outs = []
	cmd1 = 'atlasquery -a "Harvard-Oxford Cortical Structural Atlas" -c %s' %(coord, )
	cmd2 = 'atlasquery -a "Harvard-Oxford Subcortical Structural Atlas" -c %s' %(coord, )
	output1 = getoutput(cmd1)
	output2 = getoutput(cmd2)
	output1 = output1.split('br>')[1]
	output2 = output2.split('br>')[1]
	if output1 == 'No label found!':
		pass
		#outs.append('No label found!')
	else:
		rs = output1.split(', ')
		o1 = re.split('( [0-9]*%[A-Za-z ]*)',output1)
		o1 = [x.replace(',','') for x in o1 if '%' in x]
		coord_x = float(coord.split(',')[0])
		if coord_x > 0:
			dir = 'Right '
		else:
			dir = 'Left '
		rsp = re.split('(% )', o1[0])
		o1_max = rsp[0] + rsp[1] + dir + rsp[2]
		outs.append(o1_max)
	if output2 == 'No label found!':
		pass
		#outs.append('No label found!')
	else:
		rs = output2.split(', ')
		o2 = re.split('( [0-9]*%[A-Za-z ]*)',output2)
		o2 = [x.replace(',','') for x in o2 if '%' in x]
		for r in o2:
			if (not 'Cerebral Cortex' in r) and (not 'Cerebral White Matter' in r):
				outs.append(r)
				break
	if output1 == 'No label found!' and output2 == 'No label found!':
		cmd3 = 'atlasquery -a "Cerebellar Atlas in MNI152 space after normalization with FLIRT" -c %s' %(coord, )
		output3 = getoutput(cmd3)
		output3 = output3.split('br>')[1]
		if output3 == 'No label found!':
			pass
			#outs.append('No label found!')
		else:
			rs = output3.split(', ')
			o3 = re.split('( [0-9]*%[A-Za-z ]*)',output3)
			o3 = [x.replace(',','') for x in o3 if '%' in x]
			outs.append(o3[0])
	if not outs:
		return 'No label found!'
	else:
		outs_ints = [int(x.split('%')[0]) for x in outs]
		outs_ints_max = outs_ints.index(max(outs_ints))
		outs_max = outs[outs_ints_max].split('% ')[1]
	print outs
	return outs_max

def regions_file(centers_file, output_file):
    centers = file_reader(centers_file)
    region_names = []
    for c in centers:
			coord_str = ','.join(str(x) for x in c)
			region_names.append(maxprob(coord_str))
    f = open(output_file,'w')
    for r in region_names:
			f.write(r + '\n')
    f.close()

def abbrevs_file(regions_file, output_file):
    region_names = file_reader(regions_file, True)
    abbrevs = []
    harvox_region_names = {
        'Brain-Stem': 'Bstm',
        'Left Accumbens': 'LAcmb',
        'Left Amygdala': 'LAmyg',
        'Left Angular Gyrus': 'LAng',
        'Left Caudate': 'LCdt',
        'Left Central Opercular Cortex': 'LCOprc',
        'Left Cingulate Gyrus anterior division': 'LACC',
        'Left Cingulate Gyrus posterior division': 'LPCC',
        'Left Crus I': 'LCrbC1',
        'Left Crus II': 'LCrbC2',
        'Left Cuneal Cortex': 'LCun',
        'Left Frontal Medial Cortex': 'LMPFC',
        'Left Frontal Orbital Cortex': 'LOFC',
        'Left Frontal Pole': 'LFP',
        "Left Heschl's Gyrus (includes H1 and H2)": 'LHes',
        'Left Hippocampus': 'LHip',
        'Left I-IV': 'LCrb14',
        'Left IX': 'LCrb9',
        'Left Inferior Frontal Gyrus pars opercularis': 'LIFGpo',
        'Left Inferior Frontal Gyrus pars triangularis': 'LIFGpt',
        'Left Inferior Temporal Gyrus anterior division': 'LITa',
        'Left Inferior Temporal Gyrus posterior division': 'LITp',
        'Left Inferior Temporal Gyrus temporooccipital part': 'LITto',
        'Left Insular Cortex': 'LIns',
        'Left Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': 'LSMA',
        'Left Lateral Occipital Cortex inferior division': 'LLOcci',
        'Left Lateral Occipital Cortex superior division': 'LLOccs',
        'Left Lingual Gyrus': 'LLing',
        'Left Middle Frontal Gyrus': 'LMFG',
        'Left Middle Temporal Gyrus anterior division': 'LMTa',
        'Left Middle Temporal Gyrus posterior division': 'LMTp',
        'Left Middle Temporal Gyrus temporooccipital part': 'LMTto',
        'Left Occipital Fusiform Gyrus': 'LOFus',
        'Left Occipital Pole': 'LOP',
        'Left Pallidum': 'LGP',
        'Left Paracingulate Gyrus': 'LPCing',
        'Left Parahippocampal Gyrus anterior division': 'LPHGa',
        'Left Parahippocampal Gyrus posterior division': 'LPHGp',
        'Left Parietal Operculum Cortex': 'LPOprc',
        'Left Planum Polare': 'LPlPol',
        'Left Planum Temporale': 'LPlTem',
        'Left Postcentral Gyrus': 'LPreCG',
        'Left Precentral Gyrus': 'LPosCG',
        'Left Precuneous Cortex': 'LPcun',
        'Left Subcallosal Cortex': 'LSbcal',
        'Left Superior Frontal Gyrus': 'LSFG',
        'Left Superior Parietal Lobule': 'LSPL',
        'Left Superior Temporal Gyrus anterior division': 'LSTa',
        'Left Superior Temporal Gyrus posterior division': 'LSTp',
        'Left Supramarginal Gyrus anterior division': 'LSMGa',
        'Left Supramarginal Gyrus posterior division': 'LSMGp',
        'Left Temporal Fusiform Cortex anterior division': 'LTFusa',
        'Left Temporal Fusiform Cortex posterior division': 'LTFusp',
        'Left Temporal Occipital Fusiform Cortex': 'LTOFus',
        'Left Temporal Pole': 'LTP',
        'Left Thalamus': 'LTh',
        'Left V': 'LCrb5',
        'Left VI': 'LCrb6',
        'Left VIIIb': 'LCrb8b',
        'Left VIIb': 'LCrb7b',
        'No label found!': 'N/A',
        'Right Accumbens': 'RAcmb',
        'Right Amygdala': 'RAmyg',
        'Right Angular Gyrus': 'RAng',
        'Right Caudate': 'RCdt',
        'Right Central Opercular Cortex': 'RCOprc',
        'Right Cingulate Gyrus anterior division': 'RACC',
        'Right Cingulate Gyrus posterior division': 'RPCC',
        'Right Crus I': 'RCrbC1',
        'Right Crus II': 'RCrbC2',
        'Right Cuneal Cortex': 'RCun',
        'Right Frontal Operculum Cortex': 'RFOprc',
        'Right Frontal Orbital Cortex': 'ROFC',
        'Right Frontal Pole': 'RFP',
        "Right Heschl's Gyrus (includes H1 and H2)": 'RHes',
        'Right Hippocampus': 'RHip',
        'Right IX': 'RCrb9',
        'Right Inferior Frontal Gyrus pars opercularis': 'RIFGpo',
        'Right Inferior Temporal Gyrus anterior division': 'RITa',
        'Right Inferior Temporal Gyrus posterior division': 'RITp',
        'Right Inferior Temporal Gyrus temporooccipital part': 'RITto',
        'Right Insular Cortex': 'RIns',
        'Right Intracalcarine Cortex': 'RCalc',
        'Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': 'RSMA',
        'Right Lateral Occipital Cortex inferior division': 'RLOcci',
        'Right Lateral Occipital Cortex superior division': 'RLOccs',
        'Right Lingual Gyrus': 'RLing',
        'Right Middle Frontal Gyrus': 'RMFG',
        'Right Middle Temporal Gyrus anterior division': 'RMTa',
        'Right Middle Temporal Gyrus posterior division': 'RMTp',
        'Right Middle Temporal Gyrus temporooccipital part': 'RMTto',
        'Right Occipital Fusiform Gyrus': 'ROFus',
        'Right Occipital Pole': 'ROP',
        'Right Pallidum': 'RGP',
        'Right Paracingulate Gyrus': 'RPCing',
        'Right Parahippocampal Gyrus anterior division': 'RPHGa',
        'Right Parahippocampal Gyrus posterior division': 'RPHGp',
        'Right Planum Polare': 'RPlPol',
        'Right Planum Temporale': 'RPlTem',
        'Right Postcentral Gyrus': 'RPosCG',
        'Right Precentral Gyrus': 'RPreCG',
        'Right Precuneous Cortex': 'RPcun',
        'Right Putamen': 'RPut',
        'Right Subcallosal Cortex': 'RSbcal',
        'Right Superior Frontal Gyrus': 'RSFG',
        'Right Superior Parietal Lobule': 'RSPL',
        'Right Superior Temporal Gyrus anterior division': 'RSTa',
        'Right Superior Temporal Gyrus posterior division': 'RSTp',
        'Right Supramarginal Gyrus anterior division': 'RSMGa',
        'Right Supramarginal Gyrus posterior division': 'RSMGp',
        'Right Temporal Fusiform Cortex anterior division': 'RTFusa',
        'Right Temporal Fusiform Cortex posterior division': 'RTFusp',
        'Right Temporal Occipital Fusiform Cortex': 'RTOFus',
        'Right Temporal Pole': 'RTP',
        'Right Thalamus': 'RTh',
        'Right V': 'RCrb5',
        'Right VI': 'RCrb6',
        'Right VIIIa': 'RCrb8a',
        'Right VIIIb': 'RCrb8b',
        'Right VIIb': 'RCrb7b',
        'Right X': 'RCrb10',
        'Vermis VIIIa': 'CrbVrm',
    }
    for r in region_names:
        if r in harvox_region_names:
            abbrevs.append(harvox_region_names[r])
        else:
            abbrevs.append(''.join(x[0] for x in r.split()))
    f = open(output_file,'w')
    for n in abbrevs:
        f.write(n + '\n')
    f.close()
