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
Please modify the `root` to the file address under your download
Then you will get the following directory

    ...
    ├── v1
        └── scans
            ├── ...
            ├── 2t7WUuJeko7
            └── 5LpN3gDmAk7
                └──5LpN3gDmAk7
                   └──region_segmentations
                      ├── ...
                      ├── region0.fsegs.json
                      ├── region0.ply
                      ├── region0.semseg.json
                      └── region0.vsegs.json
                
Next, we simplify the grid through `graph.py` to obtain the processed pt data.

        python3 graph.py \
        --in_path /home/jin/matterport/v1/scans \
        --out_path data/matterport/ \
        --level_params 0.04 30 30 30 \
        --train \
        --qem \
        --dataset matterport 

    
