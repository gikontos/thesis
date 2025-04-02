import pandas as pd
from importlib import resources as impresources
from pathlib import Path
from classes.data import Data
from classes.annotation import Annotation


def load_all_data(mods, tsv_file=None):
    data_path = Path('../data')       # path to dataset
    
    selected_subjects = None
    if tsv_file:
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=['subject'])  # No header, so set column name
        selected_subjects = set(df['subject'].astype(str))  # Convert to a set

    ## Build recordings list:
    sub_list = [x for x in data_path.glob("sub*")]
    recordings = [[x.name, xx.name.split('_')[-2]] for x in sub_list for xx in (x / 'ses-01' / 'eeg').glob("*edf")]

    # filter recordings to choose only recordings from certain patient:
    #recordings = [x for x in recordings if 'sub-001' in x[0]]
    if selected_subjects:
        recordings = [x for x in recordings if x[0] in selected_subjects]
        
    data = list()
    annotations = list()

    prev = ''
    counter = 0
    for rec in recordings:
        #print(rec[0] + ' ' + rec[1])
        rec_data = Data.loadData(data_path.as_posix(), rec, modalities=mods)
        rec_annotations = Annotation.loadAnnotation(data_path.as_posix(), rec)
        if prev!=rec[0]: #start of code useful for sub
            print(counter)
            print(rec[0])
            #counter = 0
        counter = counter+1
        prev = rec[0] #end
        data.append(rec_data)
        annotations.append(rec_annotations)
    print(counter) #sub
    return data, annotations
