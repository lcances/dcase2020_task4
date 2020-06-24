# dcase2020

## Requirement

##### 1. Clone repo & installation
Install the repo as a local pip package 
```Bash
conda activate myenv

git clone https://github.com/leocances/dcase2020_task4.git
cd ../dcase2020_task4
pip install -e .
```

or

```bash
conda activate myenv
pip install git+https://github.com/leocances/dcase2020_task4.git
```

##### 2. Environement
```Bash
conda create -n dcase2020 python=3 pip
conda activate dcase2020

conda install pytorch
conda install h5py
conda install pandas
conda install tensorboard
conda install tqdm

conda install pillow       # for the augmentation_utils package
conda install scikit-image # ...

conda install -c conda-forge librosa
conda install -c conda-forge torchvision

pip install advertorch
pip install torchsummaryX
pip install dcase_util
pip install sed_eval

pip install git+https://github.com/leocances/dcase2020.git # <-- dcase2020 task 4 separate dataset management
pip install git+https://github.com/leocances/pytorch_metrics.git # <-- personnal pytorch metrics functions
pip install git+https://github.com/leocances/augmentation_utils.git # <-- personnal audio & image augmentation functions
pip install -i https://test.pypi.org/simple/ aeseg # for automatique segmentation optimization

# Extra package for jupyterlab (and running the notebooks smoothly)
conda install jupyterlab
conda install -c conda-forge ipywidgets
conda install -c conda-forge nodejs
```

##### 3. Download the dataset
visit: https://github.com/turpaultn/DESED/tree/master/real

## 1 - Minimun change on the DESED dataset architecture. (for easier manipulation)
For a more homogeneous architecture of the metadata file, and therefore a easier manipulation in the
datasetManager class, the sole file *soundscapes.tsv* is rename *synthetic20.tsv* and move one level above

```bash
cd DESED/dataset/metadata/train
mv synthetic20/soundscapes.tsv synthetic20.tsv
rmdir synthetic20/

cd ../../audio/train
mv synthetic20/soundscapes/* synthetic20
rmdir synthetic20/soundscapes
```

## 2 - Create the HDF file
Extract the raw-audio using librosa and store the dataset into a HDF file.

To remove DESED or FUSS, remove the `+DESED` or `+FUSS`

| :warning: final hdf file is 65 Go without compression. See below example with compression |
| --- |

```bash
conda activate dcase2020
cd standalone

python move_to_hdf.py -sr 22050 -l 10 -a ../dataset --num_workers 8 +DESED +FUSS
```

With compression. All h5py compression are supported.

| algorytms | exec time | final size | command line argument |
| --------- | --------- | ---------- | --------------------- |
| no        | ~15 min   | 65 Go      | ` `                   |
| LZF       | ~20 min   | 57 Go      | `--compression lzf`   |
| GZIP      | ~60 min   | 49 Go      | `--compression gzip`  |

# 3 - Reproduction results_bool

| :warning: Please be sure that you have follow step 1 and 2  |
| --- |


# Fast documentation
## Using the datasetManager and create a pytorch dataset
- So far, only the DESED manager is supported. The manager allow to load
the different subset independantly and create a train / validation split.
- The DESEDDatasets take care of extracting the feature in time, applying the augmentations. the same class is used for both train and validation. It has the folowing features
  - **Data augmentation** that can be apply on the signal before extraction, or on the feature after extraction.
  - **Process safe cache**  for storing the extracted feature to calculate them only once. It greatly improve training speed. :warning: The cache is automatically deactivate when using augmentation.

By default **all the raw_audio is loaded into the memory**. to avoid that and pull raw audio from disk when needed, set
*from_disk* to True
### Loading subset and create a train / val split
```python
import ...

from dcase2020_task4 import DESEDManager

metadata_root = "../path/to/metadata"
audio_root = "../path/to/audio"

manager = DESEDManager(
    metadata_root, audio_root,
    sampling_rate = 22050,
    validation_ratio = 0.2,
    from_disk=False, # <-- change to True if do not hold into memory
    verbose = 1
    )

manager.add_subset("weak")
manager.add_subset("unlabel_in_domain")
manager.add_subset("synthetic20")

manager.split_train_validation()
```

### Creating the pytorch dataset for train and val
- Without augmentations
```python
from dcase2020_task4 import DESEDDataset

train_dataset = DESEDDataset(manager, 
    train=True, val=False, augments=[],
    weak=True, strong=True, # <-- we want both weak and strong ground truth to be outputed
    cached=True)
val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True) # <-- by default only weak ground truth will be return
```

 - With augmentation:
   - noise on signal with a target SNR of 15db and 
   - random time dropout of 30% on the mel-spectrogram
 ```python
 from dcase2020_task4 import DESEDDataset

 import augmentation_utils.signal_augmentations as signal_augmentations
 import augmentation_utils.spec_augmentations as spec_augmentations

 augments = [
     signal_augmentations.Noise(0.5, target_snr=15),
     spec_augmentations.RandomTimeDropout(0.5, dropout=0.3)
 ]

 train_dataset = DESEDDataset(manager, train=True, val=False, augments=augments, cached=True) # <-- cache is automatically deactivate
 val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True)
 ```


## dataset organisation
- DESED
    - dataset
        - audio
            - train
            - validation
        - metadata
            - train
            - validation
        - missing_files
- FUSS
    - fsd_data
        - train
        - validation
        - eval
    - rir_data
        - train
        - validation
        - eval
    - ssdata
        - train
        - validation
        - eval
    - ssdata_reverb
        - train
        - validation
        - eval

