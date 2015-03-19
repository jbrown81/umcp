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
import time
from optparse import OptionParser
import core
import timeseries

def main():
    usage = "usage: run_timeseries.py -f <4d_nii_file> -m <input_masks_file> -o <output_prefix> [options]"
    parser = OptionParser(usage)
    parser.add_option("-f", "--func", action="store", type="string", dest="funcfile",
                      help="read 4D BOLD fMRI data from FILENAME.nii")
    parser.add_option("-m", "--masks", action="store", type="string", dest="masksfile",
                      help="read mask filenames stored on separate lines in FILENAME.txt")
    parser.add_option("-o", "--out", action="store", type="string", dest="output",
                      help="output file prefix")

    parser.add_option("-c", "--corr", action="store_true", dest="corr",
                      help="calculate correlation matrix between all masks")
    parser.add_option("-p", "--pcorr", action="store_true", dest="pcorr",
                      help="calculate partial correlation matrix between all masks")
    parser.add_option("-v", "--cov", action="store_true", dest="cov",
                      help="calculate covariance matrix between all masks")

    parser.add_option("--scrub", action="store", type="string", dest="scrubfile",
                      help="optional: include one column file with 1 for TRs to exclude, 0 for TRs to include")

    (options, args) = parser.parse_args()

    start = time.time()

    if not options.funcfile:
        print "Must specify input .nii file"
    
    if options.masksfile:
        masks_files = core.file_reader(options.masksfile, True)
    else:
        print "Must specify input mask list .txt file"
        
    if options.output == None:
        print "Must specify output file prefix"

    # Get connectivity matrix for all masks in list
    if options.scrubfile:
        if options.corr:
            print "timeseries.mask_funcconnec_matrix(%s, %s, %s, scrub_trs_file=%s)" % (options.funcfile, options.masksfile, options.output, options.scrubfile)
            timeseries.mask_funcconnec_matrix(options.funcfile, masks_files, options.output, scrub_trs_file=scrub_trs_file)
        elif options.pcorr:
            print "timeseries.mask_funcconnec_matrix(%s, %s, %s, partial=True, scrub_trs_file=%s)" % (options.funcfile, options.masksfile, options.output, options.scrubfile)
            timeseries.mask_funcconnec_matrix(options.funcfile, masks_files, options.output, partial=True, scrub_trs_file=scrub_trs_file)
        elif options.cov:
            print "timeseries.mask_funcconnec_matrix(%s, %s, %s, cov=True, scrub_trs_file=%s)" % (options.funcfile, options.masksfile, options.output, options.scrubfile)
            timeseries.mask_funcconnec_matrix(options.funcfile, masks_files, options.output, cov=True, scrub_trs_file=scrub_trs_file)
    else:
        if options.corr:
            print "timeseries.mask_funcconnec_matrix(%s, %s, %s)" % (options.funcfile, options.masksfile, options.output)
            timeseries.mask_funcconnec_matrix(options.funcfile, masks_files, options.output)
        elif options.pcorr:
            print "timeseries.mask_funcconnec_matrix(%s, %s, %s, partial=True)" % (options.funcfile, options.masksfile, options.output)
            timeseries.mask_funcconnec_matrix(options.funcfile, masks_files, options.output, partial=True)
        elif options.cov:
            print "timeseries.mask_funcconnec_matrix(%s, %s, %s, cov=True)" % (options.funcfile, options.masksfile, options.output)
            timeseries.mask_funcconnec_matrix(options.funcfile, masks_files, options.output, cov=True)

    elapsed = time.time() - start
    print "Took %s seconds to run" % elapsed

if __name__ == "__main__":
    main()