Prepare Matterport3D data
==
Download data
---
Download the Matterport3D data [here](https://niessner.github.io/Matterport/).
Send [MP_ TOS.pdf](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) to matterport3d@googlegroups.com , get the script file of the downloaded dataset.
After you get the download method from the mailbox, you can run the following command

    python2.7 download_mp.py -o [download path] --type region_segmentations
    
If you use python3, pay attention to changing the expression of print, and change urllib to urllib.request
For example, `urllib.request.urlopen` and `urllib.request.urlretrieve`
After downloading the dataset, you will get the following directory

    ...
    ├── v1
        └── scans
            ├── ...
            ├── 2t7WUuJeko7
            └── 5LpN3gDmAk7
                └── region_segmentations.zip     
    
We decompress files in bulk through `unzipmp.py`
