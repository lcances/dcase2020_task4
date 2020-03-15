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