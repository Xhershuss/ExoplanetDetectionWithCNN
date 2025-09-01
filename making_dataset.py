import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lightkurve as lk
from sklearn.model_selection import train_test_split

import warnings


warnings.filterwarnings('ignore', category=lk.LightkurveWarning)


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
#print(script_dir)


exoplanet_dataset_path = rf"{script_dir}\predataset\exoplanet_archive.csv"





exoplanet_dataset = pd.read_csv(exoplanet_dataset_path,comment="#")

#print(exoplanet_dataset.head(40))
counts = exoplanet_dataset['koi_disposition'].value_counts()
print(
    f"Confirmed Exoplanets   {counts.get('CONFIRMED', 0)}\n"
    f"False Positive Exoplanets {counts.get('FALSE POSITIVE', 0)}\n"
    f"Candidate Exoplanets   {counts.get('CANDIDATE', 0)}"
)
#Extracting Candidate datas
mask = exoplanet_dataset["koi_disposition"].isin(["CONFIRMED","FALSE POSITIVE"])
masked_dataset = exoplanet_dataset[mask]
#masked_dataset = exoplanet_dataset[mask].iloc[0:20]

#print(masked_dataset.head(40))

#print("flux listesi\n",lc.flux, "time listesi",lc.time)

#print(  f"2 flux datası arası gün süre (gün) {((lc.time[1])-(lc.time[0])).jd} gün\n"  f"2 flux datası arası gün süre (dk)  {(((lc.time[1])-(lc.time[0])).jd)*24*60} dk")


k = 0
rows = []

for _,row in masked_dataset.iterrows():  
    
    kepid = row['kepid'] 
    disposition= row['koi_disposition'] 
    
    try:
        lc = (lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")
                .download()
                .normalize()
                .remove_outliers(sigma=5)
                )
    except Exception as e:
        print(f"{kepid} için veri indirme hatası: {e}")
        continue

    temp = {
        'planet_id': kepid,
        'label': 1 if disposition == 'CONFIRMED' else 0 
    }
    
    for i,flux_value in enumerate(lc.flux.value,start= 0):
        temp[f"flux{i+1}"] = flux_value


    rows.append(temp)

    print(f"{k}.gezegen  {kepid} başarı ile eklendi")
    k+=1




def sepearte_dataset_without_validation(dataset):
    
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=0.2,        
        random_state=42,     
        stratify=dataset['label'] 
    )


    print(f"---train dataset---\n"
        f"{train_dataset.head(20)}\n"
        f"labeled 1 : {train_dataset['label'].value_counts().get(1, 0)}\n"
        f"labeled 0 : {train_dataset['label'].value_counts().get(0, 0)}\n\n"
        f"---test dataset---\n"
        f"{test_dataset.head(20)}\n"
        f"labeled 1 : {test_dataset['label'].value_counts().get(1, 0)}\n"
        f"labeled 0 : {test_dataset['label'].value_counts().get(0, 0)}")

    os.makedirs(f'{script_dir}/Dataset/without_validation',exist_ok=True)
    train_dataset.to_csv(f'{script_dir}/Dataset/without_validation/train_dataset.csv', index=False)
    test_dataset.to_csv(f'{script_dir}/Dataset/without_validation/test_dataset.csv', index=False)



def sepearte_dataset_with_validation(dataset):
    
    train_values, test_dataset = train_test_split(
        dataset,
        test_size=0.15,        
        random_state=42,     
        stratify=dataset['label'] 
    )

    train_dataset, validation_dataset = train_test_split(
            train_values,
            test_size=0.176 ,       
            random_state=42,     
            stratify=train_values['label'] 
        )



    print(
        f"---train dataset---\n"
        f"{train_dataset.head(20)}\n"
        f"labeled 1 : {train_dataset['label'].value_counts().get(1, 0)}\n"
        f"labeled 0 : {train_dataset['label'].value_counts().get(0, 0)}\n\n"
        f"---test dataset---\n"
        f"{test_dataset.head(20)}\n"
        f"labeled 1 : {test_dataset['label'].value_counts().get(1, 0)}\n"
        f"labeled 0 : {test_dataset['label'].value_counts().get(0, 0)}\n"
        f"---validation dataset---\n"
        f"{validation_dataset.head(20)}\n"
        f"labeled 1 : {validation_dataset['label'].value_counts().get(1, 0)}\n"
        f"labeled 0 : {validation_dataset['label'].value_counts().get(0, 0)}\n")



    os.makedirs(f'{script_dir}/Dataset/with_validation',exist_ok=True)
    train_dataset.to_csv(f'{script_dir}/Dataset/with_validation/train_dataset.csv', index=False)
    test_dataset.to_csv(f'{script_dir}/Dataset/with_validation/test_dataset.csv', index=False)
    validation_dataset.to_csv(f'{script_dir}/Dataset/with_validation/validation_dataset.csv', index=False)









dataset_with_nans = pd.DataFrame(rows)


print(dataset_with_nans)


planet_id_columns =dataset_with_nans['planet_id'] 
label_columns= dataset_with_nans['label']
flux_columns= [c for c in dataset_with_nans.columns if c.startswith('flux')]


X_raw = dataset_with_nans[flux_columns].to_numpy(dtype=float)
max_len = 40000


X_fixed = X_raw[:, :max_len]
X_fixed[np.isnan(X_fixed)] = -1

flux40k_cols = [f"flux{i+1}" for i in range(max_len)]

dataset = pd.DataFrame(X_fixed, columns=flux40k_cols)
dataset.insert(0, "label", dataset_with_nans["label"].values)
dataset.insert(0, "planet_id", dataset_with_nans["planet_id"].values)



sepearte_dataset_without_validation(dataset)
sepearte_dataset_with_validation(dataset)    