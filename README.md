# dcase2020

## Requirement
```Bash
conda create -n dcase2020 python=3 pip
conda activate dcase2020

conda install pytorch
conda install h5py
conda install pandas

pip install librosa
pip install tqdm
```

## Minimun change on the DESED dataset architecture. (for easier manipulation)
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

## Pre-processing
Extract the raw-audio using librosa and store the dataset into a HDF file.

| :warning: final hdf file is 65 Go without compression.see below example with compression |
| --- |

Without compression.
- Time with 8 workers: ~14 minutes
- Final file size: 65 Go
```bash
conda activate dcase2020
cd standalone

python move_to_hdf.py -sr 22050 -l 10 -a ../dataset --num_workers 4
```

With compression. All h5py conpression are supported.

LZF conpression:
- Time with 8 workers: ~20 minutes
- Final file size: 57 go
```bash
conda activate dcase2020
cd standalone

python move_to_hdf.py -sr 22050 -l 10 -a ../dataset --num_workers 4 --compression lzf
```

GZIP conpression:
- Time with 8 workers: ~20 minutes
- Final file size: 57 go
```bash
conda activate dcase2020
cd standalone

python move_to_hdf.py -sr 22050 -l 10 -a ../dataset --num_workers 4 --compression gzip
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