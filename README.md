# Siamese-Network for Raman spectra

'Siamese Network for Classification of Raman Spectroscopy with Inter-instrument Variation for Biological Applications'

## Creating Dataset

### Samples: 10 bacteria

* Acinetobacter baumannii (ABA)
* Citrobacter freundii (CFR)
* Escherichia coli (ECO)
* Enterococcus faecalis (EFA)
* Klebsiella oxytoca (KOX)
* Klebsiella pneumoniae (KPN)
* Stenotrophomonas maltophilia (PMA)
* Staphylococcus aureus (SAU)
* Staphylococcus hominis (SHO)
* Serratia marcescens (SMA)

### Raman spectrometer

* Hooke P300 (Hooke Instruments, Changchun, China.) 600g/mm grating and 1200g/mm grating
* Hooke R300 (Hooke Instruments, Changchun, China.) 600g/mm grating

### Raman spectra acquisition conditions

* Excitation wavelength: 532nm
* Power: 5mW
* Exposure time: 5s
* Number: 400 spectra for each bacterial with each instrument model

## Data preprocessing

The preprocessing function is in **'precess_fun.py'**.

Load data by the class in **'base.py'** and set **process_flag=True**. The steps of data preprocessing are as follows:

* Step 1：spike removal
* Step 2：baseline subtraction
* Step 3：normalization

Then, the preprocessed data can be saved in '.txt' or '.npy' files.

## Data preparation

There 3 functions in **'dataset.py'** to create csv files for data preparation.

* **'create_folder_csv'**: Create a csv file with data folder.
* **'create_fsl_ds_csv'**: Split two csv files into four csv files for training and testing.
* **'merge_fsl_csv'**: Merge two csv files into one with positive and negative pairs for SNN model Training

## Model training

### Classification task

> notebook/train_classification.ipynb
