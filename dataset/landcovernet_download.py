import os
import argparse

os.environ['MLHUB_API_KEY'] = '66690be5437c62c1d0c4b4cd7e6821a3f13957088fffc58dfd550ce8768ce699'

import urllib.parse
import re
from pathlib import Path
import itertools as it
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
# try:
from radiant_mlhub import client, get_session
# except ImportError:
    # os.system('pip install radiant_mlhub -q')
    # from radiant_mlhub import client, get_session

items_pattern = re.compile(r'^/mlhub/v1/collections/(\w+)/items/(\w+)$')
def download(item, asset_key, output_dir='./data'):
    """Downloads the given item asset by looking up that asset and then following the "href" URL."""

    # Try to get the given asset and return None if it does not exist
    asset = item.get('assets', {}).get(asset_key)
    if asset is None:
        print(f'Asset "{asset_key}" does not exist in this item')
        return None
    
    # Try to get the download URL from the asset and return None if it does not exist
    download_url = asset.get('href')
    if download_url is None:
        print(f'Asset {asset_key} does not have an "href" property, cannot download.')
        return None
    
    session = get_session()
    r = session.get(download_url, allow_redirects=True, stream=True)
    
    filename = urllib.parse.urlsplit(r.url).path.split('/')[-1]
    output_path = Path(output_dir) / filename

    
    with output_path.open('wb') as dst:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk:
                dst.write(chunk)

def download_labels_and_source(item, assets=None, output_dir='./data'):
    """Downloads all label and source imagery assets associated with a label item that match the given asset types.
    """
    
    # Follow all source links and add all assets from those
    def _get_download_args(link):
        # Get the item ID (last part of the link path)
        source_item_path = urllib.parse.urlsplit(link['href']).path
        source_item_collection, source_item_id = items_pattern.fullmatch(source_item_path).groups()
        source_item = client.get_collection_item(source_item_collection, source_item_id)

        source_download_dir = download_dir / 'source'
        source_download_dir.mkdir(exist_ok=True)
        
        matching_source_assets = [
            asset 
            for asset in source_item.get('assets', {}) 
            if assets is None or asset in assets
        ] 
        return [
            (source_item, asset, source_download_dir) 
            for asset in matching_source_assets
        ]

    
    download_args = []
    
    download_dir = Path(output_dir) / item['id']
    download_dir.mkdir(parents=True, exist_ok=True)
#     Down
    labels_download_dir = download_dir / 'labels'
    labels_download_dir.mkdir(exist_ok=True)

    # Download the labels assets
    matching_assets = [
        asset 
        for asset in item.get('assets', {}) 
        if assets is None or asset in assets
    ]

    for asset in matching_assets:
        download_args.append((item, asset, labels_download_dir))
        
    source_links = [link for link in item['links'] if link['rel'] == 'source']
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        for argument_batch in executor.map(_get_download_args, source_links):
            download_args += argument_batch
        
    print(f'Downloading {len(download_args)} assets...')
    with ThreadPoolExecutor(max_workers=16) as executor:
        with tqdm(total=len(download_args)) as pbar:
            for _ in executor.map(lambda triplet: download(*triplet), download_args):
                pbar.update(1)

def get_collections(no_collections):
    collection_id = 'ref_landcovernet_v1_labels'
    if(no_collections): return client.list_collection_items(collection_id, limit=no_collections)
    return client.list_collection_items(collection_id)

def download_dataset(path, assets, no_cols=1):
    collections = get_collections(no_cols)
    for item in collections: 
        download_labels_and_source(item, assets, output_dir=path)
        time.sleep(0.001)
        

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    
    args.add_argument('-o', '--outdir', default=None, type=str,
                      help='output dir', required=True)
    
    args.add_argument('-n', '--no_cols', default=None, type=str,
                      help='output dir')
    
    assets=['source_dates', 'labels', 'B02', 'B03', 'B04', 'CLD']
    args = args.parse_args()
    
    download_dataset(args.outdir,assets, args.no_cols)
    
    
        
      