import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def MakeModel(tg):
  if tg = 0:
    mod = tf.keras.Sequential()
    mod.add(tf.keras.Input(shape=(x_train.shape[1],)))
    mod.add(tf.keras.layers.Dense(1000, activation='relu'))
    mod.add(tf.keras.layers.BatchNormalization(axis=1))
    mod.add(tf.keras.layers.Dropout(0.1))
    mod.add(tf.keras.layers.Dense(1000, activation='relu'))
    mod.add(tf.keras.layers.Dense(1000, activation='relu'))
    mod.add(tf.keras.layers.Dense(1, activation='linear'))
  else:
    mod = tf.keras.Sequential()
    mod.add(tf.keras.Input(shape=(x_train.shape[1],)))
    mod.add(tf.keras.layers.Dense(100, activation='relu'))
    mod.add(tf.keras.layers.BatchNormalization(axis=1))
    mod.add(tf.keras.layers.Dropout(0.1))
    mod.add(tf.keras.layers.Dense(100, activation='relu'))
    mod.add(tf.keras.layers.Dense(100, activation='relu'))
    mod.add(tf.keras.layers.Dense(1, activation='linear'))7
  return mod



# To be modified with the dataset name
Shuffled_data_filename = "data.xlms"

# list of targets
targets = ['T (C)', 'P (kbar)']
names_targets = ['temperature', 'pressure']

#list of input ensembles
input_sections = ['only_cpx', 'cpx_and_liq']


for tg in [0, 1]:
   # select some parameters depending on the target
        if tg == 0:
            fr = 0.8
            N = 20
            fraction _of_bagging = 0.6
        else:
            fr = 0.95
            N = 100
            fraction _of_bagging = 0.8

    for in_s in [0, 1]:

        # load the data
        df = pd.read_excel(Shuffled_data_filename)

        #define some variables to identify the experiment
        target = targets[tg]
        names_target = names_targets[tg]
        sect = input_sections[in_s]

        # drop the excel indexes column
        df = df.drop(columns=df.columns[0])
        col = df.columns

        # identify the columns
        index_col = [col[i] for i in range(0, 1)]
        target_col = [col[i] for i in range(1, 3)]
        liq_col = [col[i] for i in range(3, 11)]
        cpx_col = [col[i] for i in range(11, 22)]
        # check columns:
        boolean_col = [col[i] for i in range(22, 24)]

        # Drop the liq columns for only_cpx case, remove rows with False in the check columns and the ones with Nan values
        if sect == 'only_cpx':
            df1 = df.drop(df[df[boolean_col[0]] == False].index)
            df1 = df1.drop(columns=liq_col)
        elif sect == 'cpx_and_liq':
            df1 = df.drop(df[df[boolean_col[1]] == False].index)
        df1 = df1.dropna(axis=0, how='any')
        
        # remove indexes. they can be saved before this point
        df1 = df1.drop(columns=index_col)

        # split targets and inputs
        split = np.split(df1, [1, 2], axis=1)
        ys = split[tg]
        xs = split[2].drop(split[2].columns[-3:], axis=1)

        
        x_split = np.split(xs, [int(len(xs) * fr)], axis=0)
        y_split = np.split(ys, [int(len(ys) * fr)], axis=0)
        x_train = x_split[0]
        x_test = x_split[1]
        y_train = y_split[0]
        y_test = y_split[1]

        # normalize the daset according to the train set
        array_max = y_train.max(axis=0).values
        y_test = y_test.div(array_max)
        y_train = y_train.div(array_max)
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')

        x_train['ascending_index'] = list(np.arange(0, x_train.shape[0]))
        ascending_index = x_train['ascending_index']
        x_train = x_train.drop(columns=['ascending_index'])

        count = np.zeros(x_train.shape[0])
        results = np.zeros((N, x_train.shape[0]))

        for e in range(N):
          
            # define a NN model depending on the target, print the summary
            mod = MakeModel(tg)
            print(mod.summary())

            # sorting the element for the itaration of the bootstrap procedure
            x_bag_ind = np.random.choice(x_train.index, int(x_train.shape[0] * fraction _of_bagging))
            x_bag = x_train.loc[x_bag_ind]
            y_bag = y_train.loc[x_bag_ind]

            x_val_ind = list(set(list(x_train.index)) - set(list(x_bag_ind)))
            x_val = x_train.loc[x_val_ind]

            # compile and fit the model
            mod.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005),
                        loss='mse',
                        run_eagerly=False)
            mod.fit(x_bag, y_bag,
                    batch_size=50,
                    epochs=500,
                    verbose=1
                    )

            # save the result and the variables
            eva = mod(x_val.values)
            results[e, ascending_index.loc[x_val_ind]] = eva[:, 0]
            count[ascending_index.loc[x_val_ind]] += 1

            mod.save("mod_" + names_target + '_' + sect + "/Bootstrap_model_" + str(e) + '.h5')
            diz1 = {'x_val_ind': x_val_ind}
            with open('mod_' + names_target + '_' + sect + '/Local_variables' + str(e) + '.pickle',
                      'wb') as handle:
                pickle.dump(diz1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # compute the average and the std for each element of the train set
        results = results * array_max[0]

        pred_mean = np.sum(results, axis=0) / count
        pred_std = []
        for j in range(results.shape[1]):
            l = []
            for i in range(results.shape[0]):
                if results[i, j] != 0:
                    l.append(results[i, j])
            pred_std.append(np.std(l))
        pred_std = np.array(pred_std)

        # save global variables
        diz = {'N': N, 'ascending_index': ascending_index, 'array_max': array_max, 'pred_mean': pred_mean,
               'pred_std': pred_std, 'x_train': x_train, 'x_test': x_test, 'y_train': y_train,
               'y_test': y_test, 'count': count, 'in_s': in_s}

        with open('mod_' + names_target + '_' + sect + '/Global_variable.pickle', 'wb') as handle:
            pickle.dump(diz, handle, protocol=pickle.HIGHEST_PROTOCOL)
