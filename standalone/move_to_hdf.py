import librosa
import os
import h5py
import tqdm
import numpy as np
from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser(prefix_chars="-+")
parser.add_argument("-sr", "--sampling_rate", default=22050, type=int)
parser.add_argument("-l", "--length", default=10, type=int)
parser.add_argument("-a", "--audio_root", default="../dataset/" , type=str)
parser.add_argument("-n", "--name", default="dcase2020_dataset", type=str)
parser.add_argument("+DESED", action="store_true", default=False)
parser.add_argument("+FUSS", action="store_true", default=False)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--compression", default=None, type=str)
parser.add_argument("--chunk_size", default=100, type=int, help="Limit the number of file store in memory before writing to hdf")
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

DESED_directories = [
    os.path.join("DESED", "dataset", "audio", "train", "weak"),
    os.path.join("DESED", "dataset", "audio", "train", "unlabel_in_domain"), # TODO this directory contain 14 Go of file. So far it can lead to memory error with computer equiped with less than 16 Go
    os.path.join("DESED", "dataset", "audio", "train", "synthetic20"),
    os.path.join("DESED", "dataset", "audio", "validation"),
]

FUSS_directories = [
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

# add the directories from DESED or / and FUSS depending on the user choice
directory_to_load = []
if args.DESED:
    directory_to_load += DESED_directories
if args.FUSS:
    directory_to_load += FUSS_directories


for directory in directory_to_load:
    print("loading : %s" % directory)
    folder_path = os.path.join("../dataset", directory)

    # prepare the group information and corresponding dataset
    print("folder_path: ", folder_path)
    file_list = os.listdir(folder_path)
    file_list = [name for name in file_list if name not in to_remove and name[-3:] == "wav"]

    # for every file of the directory, extract the raw audio
    raw_audios = []
    workers = Pool(args.num_workers)
    folder_path_duplicate = [folder_path] * len(file_list)

    # Create the HDF group and prepare the final size
    dataset_shape = (len(file_list), SR * LENGTH)

    hdf_fold = hdf.create_group(folder_path)
    hdf_fold.create_dataset("data", dataset_shape, compression=args.compression)
    hdf_fold.create_dataset("filenames", (len(file_list), ), dtype=h5py.special_dtype(vlen=str))
    for i in range(len(file_list)):
        hdf_fold["filenames"][i] = file_list[i]

    for index in tqdm.tqdm(range(0, dataset_shape[0] - args.chunk_size, args.chunk_size)):

        results = workers.starmap(load_file, zip(folder_path_duplicate[index:index+args.chunk_size], file_list[index:index+args.chunk_size]))

        # add the chunk to the hdf file
        hdf_fold["data"][index:index+args.chunk_size] = results

    workers.close()
    workers.join()

hdf.close()
