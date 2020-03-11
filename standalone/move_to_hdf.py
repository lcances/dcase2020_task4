import h5py
import librosa
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--sampling_rate", default=22050, type=int)
parser.add_argument("-l", "--length", default=10, type=int)
parser.add_argument("-a", "--audio_root", required=True, type=str)
parser.add_argument("-n", "--name", default="dcase2020_dataset", type=str)
args = parser.parse_args()

SR = args.sampling_rate
LENGTH = args.length

# Create the HDF file
hdf_path = os.path.join(args.audio_root, "%s_%s.hdf5" % (args.name, args.sampling_rate))
print("creating hdf file to : %s" % hdf_path)
hdf = h5py.File(hdf_path, 'w')

def load_files(folder_path, f):
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

