
import librosa
import os
import h5py
import tqdm
import numpy as np
from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--sampling_rate", default=22050, type=int)
parser.add_argument("-l", "--length", default=10, type=int)
parser.add_argument("-a", "--audio_root", required=True, type=str)
parser.add_argument("-n", "--name", default="dcase2020_dataset", type=str)
parser.add_argument("--num_workers", default=1, type=int)
args = parser.parse_args()

SR = args.sampling_rate
LENGTH = args.length

# Create the HDF file
hdf_path = os.path.join(args.audio_root, "%s_%s.hdf5" % (args.name, args.sampling_rate))
print("creating hdf file to : %s" % hdf_path)
hdf = h5py.File(hdf_path, 'w')

def load_file(folder_path, f):
    path = os.path.join(folder_path, f)
    raw, sr = librosa.load(path, sr=SR, res_type="kaiser_fast")

    # padding
    if len(raw) < SR * LENGTH:
        missing = (SR * LENGTH) - len(raw)
        raw = np.concatenate((raw, [0] * missing))

    # cropping
    elif len(raw) > SR * LENGTH:
        raw = raw[:SR*LENGTH]

    return raw

to_remove = [".DS_Store"]

directory_to_load = [
    os.path.join("DESED", "dataset", "audio", "train", "weak"),
    os.path.join("DESED", "dataset", "audio", "train", "unlabel_in_domain"),
    os.path.join("DESED", "dataset", "audio", "train", "synthetic20", "soundscapes"),
    os.path.join("DESED", "dataset", "audio", "validation"),

    os.path.join("FUSS", "fsd_data", "train"),
    os.path.join("FUSS", "fsd_data", "validation"),
    os.path.join("FUSS", "fsd_data", "eval"),

    os.path.join("FUSS", "rir_data", "train"),
    os.path.join("FUSS", "rir_data", "validation"),
    os.path.join("FUSS", "rir_data", "eval"),

    os.path.join("FUSS", "ssdata", "train"),
    os.path.join("FUSS", "ssdata", "validation"),
    os.path.join("FUSS", "ssdata", "eval"),

    os.path.join("FUSS", "ssdata_reverb", "train"),
    os.path.join("FUSS", "ssdata_reverb", "validation"),
    os.path.join("FUSS", "ssdata_reverb", "eval"),
]

for directory in tqdm.tqdm(directory_to_load):
    print("loading : %s" % directory)
    folder_path = os.path.join("../dataset", directory)

    # prepare the group information and coresponding dataset
    file_list = os.listdir(folder_path)
    file_list = [name for name in file_list if name not in to_remove and name[-3:] == "wav"]
    dataset_shape = (len(file_list), SR*LENGTH)

    # for every file of the directory, extract the raw audio
    raw_audios = []
    workers = Pool(args.num_workers)
    folder_path_duplicate = [folder_path] * len(file_list)

    results = workers.starmap(load_file, zip(folder_path_duplicate, file_list))

    results = np.array(results)
    file_list = np.array(file_list)

    # Create the hdf group and write the dataset
    print("folder_path : ", folder_path)
    hdf_fold = hdf.create_group(folder_path)
    hdf_fold.create_dataset("data", data=results)
    hdf_fold.create_dataset("filenames", (len(file_list), ), dtype=h5py.special_dtype(vlen=str))
    for i in range(len(file_list)):
        hdf_fold["filenames"][i] = file_list[i]

    workers.close()
    workers.join()

hdf.close()
