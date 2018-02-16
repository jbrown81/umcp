#!/opt/local/bin/python

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
import time
from optparse import OptionParser
import core
import tracks

def main():
    usage = "usage: run_tracks.py -t <input_tracks> -m <input_masks> -o <output_prefix> [options]"
    parser = OptionParser(usage)
    parser.add_option("-t", "--tracks", action="store", type="string", dest="tracksfile",
                      help="read track data from Diffusion Toolkit FILENAME.trk or DSI Studio FILENAME.txt")
    parser.add_option("-m", "--masks", action="store", type="string", dest="masksfile",
                      help="read mask filenames stored on separate lines in FILENAME.txt")
    parser.add_option("-o", "--out", action="store", type="string", dest="output",
                      help="output file prefix")
    parser.add_option("-c", "--cmat", action="store_true", dest="connectmat",
                      help="calculate connectivity matrix between all masks")
    parser.add_option("-d", "--dens", action="store_true", dest="density",
                      help="calculate number (density) of tracks intersecting each mask")
    parser.add_option("-s", "--stats", action="store_true", dest="stats",
                      help="calculate statistics for each track group")
    parser.add_option("--statimg", action="store", type="string", dest="statimage",
                      help="optional: calculate average track group value for diffusion metric (FA, MD, ...) from .nii file")
    parser.add_option("--cthrough", action="store_true", dest="cthrough",
                      help="connectmat: any part of track must hit any part of mask")
    parser.add_option("--dend", action="store_true", dest="dend",
                      help="density: either endpoint of track must hit any part of mask")    
    parser.add_option("--maskthr", type="float", dest="maskthresh",
                      help="optional: threshold value for probabilistic masks")
    parser.add_option("--lenthr", type="float", dest="lenthresh",
                      help="optional: length threshold for tracks")
    parser.add_option("--densnii", action="store_true", dest="densnii",
                      help="for density calculation, output .nii density file instead of mask hit counts in .txt file")
    parser.add_option("--dsistudio", action="store_true", dest="dsistudio",
                      help="if .trk file was generated with dsi_studio")

    (options, args) = parser.parse_args()

    start = time.time()
    filelog = open("%s_log.txt" % options.output, "w")

    if options.dsistudio:
        diffusion_toolkit = False
        dsi_studio = True
    else:
        diffusion_toolkit = True
        dsi_studio = False
        
    #if options.tracksfile.split('.')[-1] == 'txt':
    #    diffusion_toolkit = False
    #    dsi_studio = True
    #else:
    #    diffusion_toolkit = True
    #    dsi_studio = False

    if options.tracksfile:
        if diffusion_toolkit:
            print("tracks_list_mm = tracks.get_floats('%s')" % options.tracksfile)
            filelog.write("tracks_list_mm = tracks.get_floats('%s')\n" % options.tracksfile)
            tracks_list_mm = tracks.get_floats(options.tracksfile)
            print("header = tracks.get_header('%s')" % options.tracksfile)
            filelog.write("header = tracks.get_header('%s')\n" % options.tracksfile)
            header = tracks.get_header(options.tracksfile)
            
            # Convert coordinates from mm to voxel coordinates
            print("tracks_list_vox = tracks.mm_to_vox_convert(tracks_list_mm, header)")
            filelog.write("tracks_list_vox = tracks.mm_to_vox_convert(tracks_list_mm, header)\n")
            tracks_list_vox = tracks.mm_to_vox_convert(tracks_list_mm, header)
            print("tracks_list_vox_filled = tracks.add_missing_vox(tracks_list_vox)")
            filelog.write("tracks_list_vox_filled = tracks.add_missing_vox(tracks_list_vox)\n")
            tracks_list_vox_filled = tracks.add_missing_vox(tracks_list_vox)
        else:
            print("tracks_list_mm = tracks.get_floats('%s')" % options.tracksfile)
            filelog.write("tracks_list_mm = tracks.get_floats('%s')\n" % options.tracksfile)
            tracks_list_mm = tracks.get_floats(options.tracksfile)
            print("header = tracks.get_header('%s')" % options.tracksfile)
            filelog.write("header = tracks.get_header('%s')\n" % options.tracksfile)
            header = tracks.get_header(options.tracksfile)
            
            # Convert coordinates from mm to voxel coordinates
            print("tracks_list_vox = tracks.mm_to_vox_convert(tracks_list_mm, header)")
            filelog.write("tracks_list_vox = tracks.mm_to_vox_convert(tracks_list_mm, header, dsi_studio=True)\n")
            tracks_list_vox = tracks.mm_to_vox_convert(tracks_list_mm, header, dsi_studio=True)
            print("tracks_list_vox_filled = tracks.add_missing_vox(tracks_list_vox)")
            filelog.write("tracks_list_vox_filled = tracks.add_missing_vox(tracks_list_vox)\n")
            tracks_list_vox_filled = tracks.add_missing_vox(tracks_list_vox)
            
            #print "tracks_list_vox_filled = tracks.get_tracks_dsi_studio(%s)" % options.tracksfile
            #filelog.write("tracks_list_vox_filled = tracks.get_tracks_dsi_studio(%s)\n" % options.tracksfile)
            #tracks_list_vox_filled = tracks.get_tracks_dsi_studio(options.tracksfile)
    else:
        print("Must specify input .trk/.txt file")
    
    if options.masksfile:
        fin = open(options.masksfile)
        mask_list = []
        for line in fin:
            pos = line.rstrip()
            mask_list.append(pos)
        fin.close()
    else:
        print("Must specify input mask list .txt file")
        
    if options.output == None:
        print("Must specify output file prefix")
    
    if options.maskthresh == None:
        options.maskthresh = 0
    if options.cthrough == None:
        cthrough = 0
    else:
        cthrough = 1
    if options.dend == None:
        dthrough = 1
    else:
        dthrough = 0
    if options.lenthresh == None:
        options.lenthresh = 0
    if options.densnii == None:
        options.densnii = 0
    
    if options.connectmat:
        # Get connectivity matrix for all masks in list
        if options.lenthresh:
            #if dsi_studio:
            #    print "outmat,tracknums_mat=tracks.mask_connectivity_matrix_dsi(tracks_list_vox_filled,mask_list,'%s',%s,%s,tracks_list_mm,%s)" % (options.output, options.maskthresh, cthrough,options.lenthresh)
            #    filelog.write("outmat,tracknums_mat=tracks.mask_connectivity_matrix_dsi(tracks_list_vox_filled,mask_list,'%s',%s,%s,tracks_list_mm,%s)\n" % (options.output, options.maskthresh, cthrough,options.lenthresh))
            #    outmat,tracknums_mat=tracks.mask_connectivity_matrix_dsi(tracks_list_vox_filled,mask_list, options.output, options.maskthresh, cthrough,tracks_list_mm,options.lenthresh)
            #else:
                print("outmat,tracknums_mat=tracks.mask_connectivity_matrix(tracks_list_vox_filled,header,mask_list,'%s',%s,%s,tracks_list_mm,%s)" % (options.output, options.maskthresh, cthrough,options.lenthresh))
                filelog.write("outmat,tracknums_mat=tracks.mask_connectivity_matrix(tracks_list_vox_filled,header,mask_list,'%s',%s,%s,tracks_list_mm,%s)\n" % (options.output, options.maskthresh, cthrough,options.lenthresh))
                outmat,tracknums_mat=tracks.mask_connectivity_matrix(tracks_list_vox_filled,header,mask_list, options.output, options.maskthresh, cthrough,tracks_list_mm,options.lenthresh)
        else:
            #if dsi_studio:
            #    print "outmat,tracknums_mat=tracks.mask_connectivity_matrix_NEW(tracks_list_vox_filled,mask_list,'%s',%s,%s)" % (options.output, options.maskthresh, cthrough)
            #    filelog.write("outmat,tracknums_mat=tracks.mask_connectivity_matrix_NEW(tracks_list_vox_filled,mask_list,'%s',%s,%s)\n" % (options.output, options.maskthresh, cthrough))
            #    outmat,tracknums_mat=tracks.mask_connectivity_matrix_dsi(tracks_list_vox_filled,mask_list, options.output, options.maskthresh, cthrough)                
            #else:
                print("outmat,tracknums_mat=tracks.mask_connectivity_matrix(tracks_list_vox_filled,header,mask_list,'%s',%s,%s)" % (options.output, options.maskthresh, cthrough))
                filelog.write("outmat,tracknums_mat=tracks.mask_connectivity_matrix(tracks_list_vox_filled,header,mask_list,'%s',%s,%s)\n" % (options.output, options.maskthresh, cthrough))
                outmat,tracknums_mat=tracks.mask_connectivity_matrix(tracks_list_vox_filled,header,mask_list, options.output, options.maskthresh, cthrough)
        
        if options.stats:
            if options.statimage:
                print("volumemat,lengthmat,curvemat,statmat = tracks.track_stats_group(tracknums_mat,tracks_list,header,'%s',tracks_list_vox_filled,'%s')" % (options.output, options.statimage))
                filelog.write("volumemat,lengthmat,curvemat,statmat = tracks.track_stats_group(tracknums_mat,tracks_list,header,'%s',tracks_list_vox_filled,'%s')\n" % (options.output, options.statimage))
                volumemat,lengthmat,curvemat,statmat = tracks.track_stats_group(tracknums_mat,tracks_list_mm,header,options.output,tracks_list_vox_filled,options.statimage)
            else:
                print("volumemat,lengthmat,curvemat = tracks.track_stats_group(tracknums_mat,tracks_list,header,'%s')" % options.output)
                filelog.write("volumemat,lengthmat,curvemat = tracks.track_stats_group(tracknums_mat,tracks_list,header,'%s')\n" % options.output)
                volumemat,lengthmat,curvemat = tracks.track_stats_group(tracknums_mat,tracks_list_mm,header,options.output)

    if options.density:
        # Get density files for all masks in list and write as 4D .nii file
        if options.lenthresh:
            print("tracknums_list = tracks.mask_tracks(tracks_list_vox_filled,header,mask_list,%s,%s,%s,'%s',tracks_list_mm,%s)" % (options.maskthresh, dthrough, options.densnii, options.output,options.lenthresh))
            filelog.write("tracknums_list = tracks.mask_tracks(tracks_list_vox_filled,header,mask_list,%s,%s,%s,'%s',tracks_list_mm,%s)\n" % (options.maskthresh, dthrough, options.densnii, options.output,options.lenthresh))
            tracknums_list = tracks.mask_tracks(tracks_list_vox_filled,header,mask_list, options.maskthresh, dthrough, options.densnii, options.output,tracks_list_mm,options.lenthresh)
        else:
            print("tracknums_list = tracks.mask_tracks(tracks_list_vox_filled,header,mask_list,%s,%s,%s,'%s')" % (options.maskthresh, dthrough, options.densnii, options.output))
            filelog.write("tracknums_list = tracks.mask_tracks(tracks_list_vox_filled,header,mask_list,%s,%s,%s,'%s')\n" % (options.maskthresh, dthrough, options.densnii, options.output))
            tracknums_list = tracks.mask_tracks(tracks_list_vox_filled,header,mask_list, options.maskthresh, dthrough, options.densnii, options.output)
        
        if options.stats:
            if options.statimage:
                print("volumelist,lengthlist,curvelist,statlist = tracks.track_stats_list(tracknums_list,tracks_list_mm,header,'%s',tracks_list_vox_filled,'%s')" % (options.output, options.statimage))
                filelog.write("volumelist,lengthlist,curvelist,statlist = tracks.track_stats_list(tracknums_list,tracks_list_mm,header,'%s',tracks_list_vox_filled,'%s')\n" % (options.output, options.statimage))
                volumelist,lengthlist,curvelist,statlist = tracks.track_stats_list(tracknums_list,tracks_list_mm,header,options.output,tracks_list_vox_filled,options.statimage)
            else:
                print("volumelist,lengthlist,curvelist = tracks.track_stats_list(tracknums_list,tracks_list_mm,header,'%s')" % options.output)
                filelog.write("volumelist,lengthlist,curvelist = tracks.track_stats_list(tracknums_list,tracks_list_mm,header,'%s')\n" % options.output)
                volumelist,lengthlist,curvelist = tracks.track_stats_list(tracknums_list,tracks_list_mm,header,options.output)

    if options.connectmat == None and options.density == None:
        print("Must specify either -c or -d")

    elapsed = time.time() - start
    print("Took %s seconds to run" % elapsed)
    filelog.write("Took %s seconds to run" % elapsed)
    filelog.close()

if __name__ == "__main__":
    main()