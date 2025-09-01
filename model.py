import os
import pandas as pd
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from zoneinfo import ZoneInfo  

#warning ignore
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def normalize_data(X):

    if X.ndim == 3 and X.shape[-1] == 1:
        X_flat = X[..., 0]          
    elif X.ndim == 2:
        X_flat = X                 
    else:
        raise ValueError(f"Invalid X shape: {X.shape}")
    
   
    mean = X_flat.mean(axis=1, keepdims=True)    
    std  = X_flat.std (axis=1, keepdims=True) + 1e-6
    
 
    X_norm = (X_flat - mean) / std              
    return X_norm[..., np.newaxis]               


flux_size = 2000
flux_columns = [f"flux_{i}" for i in range(40000)]
flux_2k = flux_columns[:flux_size]  



train_data_path = os.path.join(script_dir, 'Dataset', 'phase_fold', 'train_data.csv')
pre_train_data = pd.read_csv(train_data_path)
train_data = (pre_train_data[["label"]+flux_2k]
        .interpolate(axis=1, limit_direction="both")
        .to_numpy())
X_train = train_data[:,1:2000]
Y_train = train_data[:,0]

test_data_path = os.path.join(script_dir, 'Dataset', 'phase_fold', 'test_data.csv')
pre_test_data = pd.read_csv(test_data_path)
test_data = (pre_test_data[["label"]+flux_2k]
        .interpolate(axis=1, limit_direction="both")
        .to_numpy())
X_test = test_data[:,1:2000]
Y_test = test_data[:,0]


val_data_path = os.path.join(script_dir, 'Dataset', 'phase_fold', 'validation_data.csv')
pre_val_data = pd.read_csv(val_data_path)
val_data = (pre_val_data[["label"]+flux_2k]
        .interpolate(axis=1, limit_direction="both")
        .to_numpy())
X_val = val_data[:,1:2000]
Y_val = val_data[:,0]

X_train = normalize_data(X_train)
X_val   = normalize_data(X_val)
X_test  = normalize_data(X_test)





def dataset_for_model(X, y, shuffle=False, batch_size=32):
  
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


batch_size = 64
train_ds = dataset_for_model(X_train, Y_train, shuffle=True, batch_size=batch_size)
val_ds   = dataset_for_model(X_val,   Y_val,   shuffle=False, batch_size=batch_size)
test_ds  = dataset_for_model(X_test,  Y_test,  shuffle=False, batch_size=batch_size)



from sklearn.utils import class_weight

weights = class_weight.compute_class_weight(

    class_weight = {0:1.0, 1:1.5} ,
    classes=np.unique(Y_train),
    y=Y_train
)


class_weight_dict = {int(cls): weight for cls, weight in zip(np.unique(Y_train), weights)}
print("Class weights:", class_weight_dict)

from tensorflow.keras import layers, models, regularizers

def build_model(input_length):
    inputs = layers.Input(shape=(input_length, 1))

   
    x = layers.Conv1D(
            16, 5, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(
            16, 5, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)


    x = layers.Conv1D(
            32, 5, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(
            32, 5, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)


    x = layers.Conv1D(
            64, 5, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(
            64, 5, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)


    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)


    x = layers.Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    x = layers.Dropout(0.5)(x)


    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="light_model")
    return model


input_length = X_train.shape[1]   

model = build_model(input_length=input_length)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', 
                                     patience=10, 
                                     mode='max', 
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', 
                                       monitor='val_auc', 
                                       save_best_only=True, 
                                       mode='max')
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    class_weight=class_weight_dict,
    callbacks=callbacks
)


results = model.evaluate(test_ds)
print(f"Test   loss = {results[0]:.4f}")
print(f"Test accuracy = {results[1]:.4%}")
print(f"Test AUC      = {results[2]:.4f}")

from sklearn.metrics import roc_curve

probs = model.predict(test_ds).flatten()
fpr, tpr, thresholds = roc_curve(Y_test, probs)

best_idx = np.argmax(tpr - fpr)
best_thresh = thresholds[best_idx]
print("Optimal threshold:", best_thresh)

y_pred_opt = (probs >= best_thresh).astype(int)

from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_opt))
print("\nClassification Report:\n", classification_report(Y_test, y_pred_opt))




now_ist = datetime.now(ZoneInfo("Europe/Istanbul"))
timestamp = now_ist.strftime("%H%M%S")

model_filename = f'{script_dir}/Models/model_ac{results[1]:.4f}_{timestamp}.keras'
model.save(model_filename)