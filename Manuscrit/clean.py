# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import glob
import os



def clean_folder(path_fld):
    files = glob.glob(path_fld + '/*.nlg')
    files += glob.glob(path_fld + '/*.nlo')
    files += glob.glob(path_fld + '/*.nls')
    files += glob.glob(path_fld + '/*.bak')
    files += glob.glob(path_fld + '/*.log')
#    files += glob.glob(path_fld + '/*.toc')
    files += glob.glob(path_fld + '/*.synctex.gz')
#    files += glob.glob(path_fld + '/*.out')
    files += glob.glob(path_fld + '/*.mtc9')
    files += glob.glob(path_fld + '/*.mtc8')
    files += glob.glob(path_fld + '/*.mtc7')
    files += glob.glob(path_fld + '/*.mtc6')
    files += glob.glob(path_fld + '/*.mtc5')
    files += glob.glob(path_fld + '/*.mtc4')
    files += glob.glob(path_fld + '/*.mtc3')
    files += glob.glob(path_fld + '/*.mtc2')
    files += glob.glob(path_fld + '/*.mtc1')
    files += glob.glob(path_fld + '/*.mtc0')
    files += glob.glob(path_fld + '/*.mtc')
    files += glob.glob(path_fld + '/*.maf')
    files += glob.glob(path_fld + '/*.dvi')
#    files += glob.glob(path_fld + '/*.aux')
#   files += glob.glob(path_fld + '/*.bbl')
    files += glob.glob(path_fld + '/*.lof')
    files += glob.glob(path_fld + '/*.lot')
    files += glob.glob(path_fld + '/*.idx')
    files += glob.glob(path_fld + '/*.blg')
    if os.path.exists(path_fld + '/build'):
        files_build = glob.glob(path_fld + '/build/*')
        files_build_pdf = glob.glob(path_fld + '/build/*.pdf')
        files += list(set(files_build) - set(files_build_pdf))
        files_build = glob.glob(path_fld + '/build/*/*')
        files_build_pdf = glob.glob(path_fld + '/build/*/*.pdf')
        files += list(set(files_build) - set(files_build_pdf))
        files_build = glob.glob(path_fld + '/build/*/*/*.*')
        files_build_pdf = glob.glob(path_fld + '/build/*/*/*.pdf')
        files += list(set(files_build) - set(files_build_pdf))
    return files


main_fld = os.path.dirname(os.path.abspath(__file__))
to_delete = clean_folder(main_fld)
for f in glob.glob(main_fld + '/Text/*'):
    to_delete += clean_folder(f)

for f in to_delete:
    os.remove(f)
