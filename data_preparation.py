import os
import pandas as pd
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


val_data_path = os.path.join(script_dir, 'Dataset', 'with_validation', 'validation_dataset.csv')
train_data_path =os.path.join(script_dir, 'Dataset', 'with_validation', 'train_dataset.csv')
test_data_path = os.path.join(script_dir, 'Dataset', 'with_validation', 'test_dataset.csv')

val_data = pd.read_csv(val_data_path)
val_flux =  val_data.iloc[:, 2:].to_numpy()
val_labels = val_data['label'].to_numpy()


train_data = pd.read_csv(train_data_path)
train_flux =  train_data.iloc[:, 2:].to_numpy()
train_labels = train_data['label'].to_numpy()


test_data = pd.read_csv(test_data_path)
test_flux =  test_data.iloc[:, 2:].to_numpy()
test_labels = test_data['label'].to_numpy()




def discard_ones(row):
    row = np.array(row)
    mask = (row > 0)

    cleaned_row = row[mask]

    return cleaned_row

def time_values(k):
    return np.arange(0, len(k))*(30/60/24) 
    

def dataset_info(*args):
    print(f"----------------------------")
    for i in range(len(args)):
        print(f"Shape : {args[i].shape}\n"
              f"{args[i][0:5]}\n"
              f"----------------------------")
            


def curve_folding(lc):
    total_time = (lc.time.max() - lc.time.min()).to_value('day')

    min_period = 0.5      
    max_period = total_time / 3  
    n_periods = 2000 
    period_grid = np.linspace(min_period, max_period, n_periods)
    bls = lc.to_periodogram(method="bls", period=period_grid, frequency_factor=200)

    best_period = bls.period_at_max_power
    epoch_time = bls.transit_time_at_max_power

    folded = lc.fold(period=best_period, epoch_time=epoch_time)
  
    return folded




def pipeline(some_flux,flag):
    X = []
    Y = []
    for idx in range(len(some_flux)):
        clean_flux = discard_ones(some_flux[idx])

        smooth_flux = savgol_filter(clean_flux, window_length=101, polyorder=2)


        lc = lk.LightCurve(flux=smooth_flux, time =time_values(clean_flux) )
        lc_flat = lc.flatten() 

       
        fold_data = curve_folding(lc_flat)
        fold_values = fold_data.flux.value
        X.append(fold_values)

        if(flag == 1):
            Y.append(val_labels[idx])
        if(flag == 2):
            Y.append(train_labels[idx])
        if(flag == 3):
            Y.append(test_labels[idx])

  
    return X,Y


def fold_to_csv():

    os.makedirs(f'{script_dir}/Dataset/phase_fold', exist_ok=True)


    X_val, Y_val = pipeline(val_flux,1)

    max_len = max(len(arr) for arr in X_val)

    X_val_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                    for arr in X_val]
    X_arr = np.vstack(X_val_padded)
    cols = [f"flux_{i}" for i in range(X_arr.shape[1])]
    df_val = pd.DataFrame(X_arr, columns=cols)
    df_val.insert(0, "label", Y_val)
    df_val.to_csv(f'{script_dir}/Dataset/phase_fold/validation_data.csv', index=False)



    X_train, Y_train = pipeline(train_flux,2)
    max_len = max(len(arr) for arr in X_train)
    X_train_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                      for arr in X_train]
    X_arr = np.vstack(X_train_padded)
    cols = [f"flux_{i}" for i in range(X_arr.shape[1])]
    df_train = pd.DataFrame(X_arr, columns=cols)
    df_train.insert(0, "label", Y_train)
    df_train.to_csv(f'{script_dir}/Dataset/phase_fold/train_data.csv', index=False)


    X_test, Y_test = pipeline(test_flux,3)
    max_len = max(len(arr) for arr in X_test)
    X_test_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                     for arr in X_test]
    X_arr = np.vstack(X_test_padded)
    cols = [f"flux_{i}" for i in range(X_arr.shape[1])]
    df_test = pd.DataFrame(X_arr, columns=cols)
    df_test.insert(0, "label", Y_test)
    df_test.to_csv(f'{script_dir}/Dataset/phase_fold/test_data.csv', index=False)






fold_to_csv()



