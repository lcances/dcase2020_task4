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