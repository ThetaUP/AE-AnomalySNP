import os
import random

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import seaborn as sns

import os
from getpass import getpass

import neptune as neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

def model(input_dim_conti, params):
    encoder_input = tf.keras.layers.Input(shape=(input_dim_conti,))

    encoder = tf.keras.layers.Dense(params['yencoded_layer_1'], activation='elu')(encoder_input)
    encoder = tf.keras.layers.Dense(params['yencoded_layer_2'], activation='elu')(encoder)
    encoder = tf.keras.layers.Dense(params['yencoded_layer_3'], activation='elu')(encoder)
    encoder = tf.keras.layers.Dense(params['yencoded_layer_4'], activation='elu')(encoder)
    encoder = tf.keras.layers.Dense(params['yencoded_layer_5'], activation='elu')(encoder)

    shared_output = tf.keras.layers.Dense(params['shared_layer'], activation='elu')(encoder)

    decoder = tf.keras.layers.Dense(params['yencoded_layer_5'], activation='elu')(shared_output)
    decoder = tf.keras.layers.Dense(params['yencoded_layer_4'], activation='elu')(decoder)
    decoder = tf.keras.layers.Dense(params['yencoded_layer_3'], activation='elu')(decoder)
    decoder = tf.keras.layers.Dense(params['yencoded_layer_2'], activation='elu')(decoder)
    decoder = tf.keras.layers.Dense(params['yencoded_layer_1'], activation='elu')(decoder)

    Ydecoder = tf.keras.layers.Dense(input_dim_conti)(decoder)

    autoencoder = tf.keras.Model(encoder_input, Ydecoder)

    opt_ad = tf.keras.optimizers.Adam(params['lr'])
    autoencoder.compile(optimizer=opt_ad, loss='mse', metrics='mae')

    return autoencoder

def tf_learn(normal_train_data, input_dim_conti, params, run):
    nb_epoch = params['epoch']
    batch_size = params['batch_size']

    neptune_callback = NeptuneCallback(run=run) 

    all_mae_history_train = []
    all_mae_history_val = []

    # --------------------------------
    all_mae_loss_history_train = []
    all_mae_loss_history_val = []

    # --------------------------------  
                                            
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, val_index) in enumerate(kf.split(normal_train_data)):
        autoencoder = model(input_dim_conti, params)

        history = autoencoder.fit(normal_train_data[train_index,0:input_dim_conti], normal_train_data[train_index,0:input_dim_conti],
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=2,
                            callbacks=[neptune_callback],
                            validation_data=(normal_train_data[val_index,0:input_dim_conti], normal_train_data[val_index,0:input_dim_conti])
                            )

        all_mae_history_train.append(history.history['mae'])
        all_mae_history_val.append(history.history['val_mae'])
        
        all_mae_loss_history_train.append(history.history['loss'])
        all_mae_loss_history_val.append(history.history['val_loss'])

        tf.keras.backend.clear_session()

    res = {'all_mae_history_train': np.mean(all_mae_history_train, axis=0), 
           'all_mae_history_val': np.mean(all_mae_history_val, axis=0), 
           'all_mae_loss_history_train': np.mean(all_mae_loss_history_train, axis=0), 
           'all_mae_loss_history_val': np.mean(all_mae_loss_history_val, axis=0),
           'all_mae_history_train_std': np.std(all_mae_history_train, axis=0), 
           'all_mae_history_val_std': np.std(all_mae_history_val, axis=0), 
           'all_mae_loss_history_train_std': np.std(all_mae_loss_history_train, axis=0), 
           'all_mae_loss_history_val_std': np.std(all_mae_loss_history_val, axis=0)

           }
    
    for epoch in range(nb_epoch): 
        run["train/all_mae_history_train"].append(res['all_mae_history_train'][epoch])
        run["val/all_mae_history_val"].append(res['all_mae_history_val'][epoch])
        run["train/all_mae_loss_history_train"].append(res['all_mae_loss_history_train'][epoch])
        run["val/all_mae_loss_history_val"].append(res['all_mae_loss_history_val'][epoch])

    return history, res

def tf_learn_final(normal_train_data, input_dim_conti, params, run):
    nb_epoch = params['epoch']
    batch_size = params['batch_size']

    directory = f'results/models/MODEL_{model_name}_PC_{input_dim_conti}'
    checkpoint_filepath = directory + '/checkpoint_{epoch:02d}.ckpt'

    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                mode='min', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, save_freq='epoch')                                              

    autoencoder = model(input_dim_conti, params)

    history = autoencoder.fit(normal_train_data[:,0:input_dim_conti], normal_train_data[:,0:input_dim_conti],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=2,
                        callbacks=[cp],
                        )

    return history

def reconstruction_error_plot(df, column, input_dim_conti, path, ylims = (None, None), title="Reconstruction error for different classes"):  
    orange, purple = '#ffa022', '#4012ff'
    groups = df.groupby('CORRECT')
    sns.set(style="white")
    sns.set_palette(sns.color_palette([orange, purple]))
    fig, ax = plt.subplots(figsize=(8,6)) 

    for name, group in groups:
        ax.plot(group.index, 
                group[column], 
                marker='o', 
                ms=3, 
                linestyle='',
                label= "Incorrect SNP" if name == 1 else "Correct SNP")
        
    ax.legend()
    plt.title(title, fontsize=16)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.ylim(top=1.0, bottom=0.0)
    plt.legend(loc='best')
    plt.savefig(path + f'Reconstruction_plot_Model_{model_name}_PC_{input_dim_conti}.png')

def result_test(autoencoder, data_test, input_dim_conti, path):
    data_test_org = pd.read_csv('data/sample_test_data.csv')

    # Predict the data:
    test_predictions = autoencoder.predict(data_test[:,0:input_dim_conti])

    # Calculate the MAE
    test_MAE = tf.keras.metrics.MAE(data_test[:,0:input_dim_conti], test_predictions[:,0:input_dim_conti])
    test_MAE = pd.Series(test_MAE, name='MAE')

    dt = pd.concat([test_MAE, data_test_org['CORRECT']], axis=1)
    dt.to_csv(path + f'Reconstruction_table_Model_{model_name}_PC_{input_dim_conti}.csv')

    # Calculate the reconstruction matrix for all PCs (AE - absolute error)
    test_AE = pd.DataFrame(data_test[:,0:input_dim_conti] - test_predictions[:,0:input_dim_conti])

    dt_AE = pd.concat([test_AE, data_test_org['CORRECT']], axis=1, ignore_index=True)

    dt_AE.columns = [f"PC: {i}" for i in range(1, input_dim_conti + 1)] + ["CORRECT"]
    dt_AE.to_csv(path + f'Errors_RAW_Reconstruction_table_Model_{model_name}_PC_{input_dim_conti}.csv', index=False)

    # Test data using MW test
    U1, p = mannwhitneyu(dt['MAE'][dt['CORRECT']==1], dt['MAE'][dt['CORRECT']==0], alternative='greater')

    # Plot the data (not necessary now):
    reconstruction_error_plot(dt, 'MAE', input_dim_conti, path=path)
    
    return U1, p

if __name__ == "__main__":

    # Set global variables
    tf.keras.utils.set_random_seed(42)
    os.environ["NEPTUNE_API_TOKEN"] = ""
    os.environ["NEPTUNE_PROJECT"] = ""
    
    data_train = pd.read_csv('sample_train_data.csv')
    data_train = np.array(data_train)
    data_test = pd.read_csv('sample_test_data.csv')
    data_test = np.array(data_test)

    model_name = 'L'
    N_PCAs = 23

    df = pd.DataFrame(columns = ['PC',
                                'Mean LOSS Train',
                                'SD LOSS Train',
                                'Mean MAE Train',
                                'SD MAE Train',
                                'Mean LOSS Val',
                                'SD LOSS Val',
                                'Mean MAE Val',
                                'SD MAE Val',
                                'MannU',
                                'P VAL',
                                'MODEL'])

    for PC in range(3, N_PCAs + 1, 2):
        input_dim_conti = PC

        np.random.seed(42)

        run = neptune.init_run(tags=["FAMD", 'GRID ALL', f'PC: {input_dim_conti}', f'MODEL: {model_name}'])

        params = {
            "epoch": 100,
            "batch_size": 242,
            "yencoded_layer_1": 35,
            "yencoded_layer_2": 30,
            "yencoded_layer_3": 25,
            "yencoded_layer_4": 20,
            "yencoded_layer_5": 15,
            "shared_layer": 10,
            "optimizer": "Adam",
            "input_dim_conti" : input_dim_conti,
            "lr" : 0.004
        }

        run["parameters"] = params

        history, res = tf_learn(data_train, input_dim_conti, params, run)

        history_final = tf_learn_final(data_train, input_dim_conti, params, run)

        autoencoder = model(input_dim_conti, params)
        latest = tf.train.latest_checkpoint(f'results/models/MODEL_{model_name}_PC_{input_dim_conti}')
        autoencoder.load_weights(latest)

        U1, p = result_test(autoencoder, data_test, input_dim_conti, path=f'results/MODEL_{model_name}/')

        df_new = pd.Series({'PC': input_dim_conti,
                      'Mean LOSS Train': res['all_mae_loss_history_train'][-1], 
                      'SD LOSS Train': res['all_mae_loss_history_train_std'][-1], 
                      'Mean MAE Train': res['all_mae_history_train'][-1], 
                      'SD MAE Train': res['all_mae_history_train_std'][-1], 
                      'Mean LOSS Val': res['all_mae_loss_history_val'][-1], 
                      'SD LOSS Val': res['all_mae_loss_history_val_std'][-1], 
                      'Mean MAE Val':res['all_mae_history_val'][-1],
                      'SD MAE Val': res['all_mae_history_val_std'][-1], 
                      'MannU': U1, 
                      'P VAL': p, 
                      'MODEL': f'MODEL: {model_name}'})

        df= pd.concat([df, df_new.to_frame().T], ignore_index=True)

    df.to_csv(f'results/MODEL_{model_name}_RES.csv', index=False)

        

# %%
