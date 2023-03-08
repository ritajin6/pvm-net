import os
from zipfile import ZipFile
import glob

root = "/home/jin/matterport/v1/scans"

file_paths = sorted([x for
                     x in glob.glob(f"{root}/*/*.zip")])

for i in range(len(file_paths)):
    with ZipFile(file_paths[i], 'r') as zip:
        outpath = os.path.split(file_paths[i])[0]
        zip.printdir()
        zip.extractall(path=outpath)
        os.remove(file_paths[i])
print("Complete decompression！")
