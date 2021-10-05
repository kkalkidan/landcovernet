# import urllib.parse
# import re
# from pathlib import Path
# import itertools as it   
from radiant_mlhub import client
import time
import os

import subprocess

def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')

os.environ['MLHUB_API_KEY'] = '66690be5437c62c1d0c4b4cd7e6821a3f13957088fffc58dfd550ce8768ce699'

collection_id = 'ref_landcovernet_v1_source'

while True:
    try:
        a = client.download_archive(collection_id, output_dir='./data', if_exists='resume')
    except:
        print("Error occurred but retrying again...")
        size = du("./data")
        if(size[-1] == 'G' and float(size[:-1]) > 81):
            break
        time.sleep(0.1)
            
    