### extracting/downloading files from dataverse using API

import os
from pyDataverse.api import NativeApi, DataAccessApi
import patoolib
import pydicom as dicom
import cv2
import numpy as np


# Working Variables
rootDir = './'
data_dir = rootDir + 'raw_data/'
data_rar_sub = data_dir + 'rar/'
data_dcm_sub = data_dir + 'dcm/'
if not os.path.isdir(data_rar_sub): os.mkdir(data_rar_sub)
if not os.path.isdir(data_dcm_sub): os.mkdir(data_dcm_sub)
image_dir = 'raw_images/'
base_url = 'https://dataverse.harvard.edu/'
api = NativeApi(base_url)
data_api = DataAccessApi(base_url)
DOI = "doi:10.7910/DVN/6ACUZJ"
dataset = api.get_dataset(DOI)
files_list = dataset.json()['data']['latestVersion']['files']


# Extract from server, convert, and store in appropriate folders
for file in files_list:

    filename = file["dataFile"]["filename"]
    if not os.path.exists(rootDir + data_dir + filename):
        file_id = file["dataFile"]["id"]
        print("File name {}, id {}".format(filename, file_id))
        response = data_api.get_datafile(file_id)
        with open(data_rar_sub + filename, "wb") as f:
            f.write(response.content)
    else:
        print('skipped', filename)

    # Unzip and store files
    patoolib.extract_archive(data_rar_sub + filename, outdir=data_dcm_sub)
    folder = data_dcm_sub + filename.split('.')[0] + '/'
    for imID in os.listdir(folder):
        img = dicom.dcmread(folder + imID).pixel_array.astype(float)
        img = np.array(img, dtype=float)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint16)
        cv2.imwrite(image_dir + imID + '.jpg', img)








