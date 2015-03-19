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

from distutils.core import setup

setup(name = 'UMCP',
      version = '1.3',
      description = 'UCLA Multimodal Connectivity Package',
      author = 'Jesse Brown',
      author_email = 'jbrown@memory.ucsf.edu',
      url = 'http://www.ccn.ucla.edu/wiki/index.php/UCLA_Multimodal_Connectivity_Package',
      py_modules = ['tracks', 'core', 'run_tracks', 'timeseries', 'pyentropy', 'run_timeseries'],
     )
