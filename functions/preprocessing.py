import pandas as pd
import numpy as np
import pickle

#### PREPROCESSING ####

def preprocessing(df):

    # drop empty columns
    #df = df.dropna(axis=1, how='all')
    df_1 = df.copy()

    # molecular weight
    mw = {'SiO2': 60.084, 'TiO2': 79.900, 'Al2O3': 101.960, 'Cr2O3': 151.990, 'FeO': 71.846, 'MnO': 70.937,
          'NiO': 74.699, 'MgO': 40.304, 'CaO': 56.079, 'Na2O': 61.979, 'K2O': 94.196, 'Fe2O3': 159.69}
    mw_arr = np.array(list(mw.values()))

    # moles of oxides
    mole_ox = np.array([1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2])

    # compute the correction factor and multiply by it
    df_1[df.columns[4:-1]] = df[df.columns[4:-1]] / mw_arr[:-1] * mole_ox
    corr_fact = 4 / df_1[df.columns[4:-1]].sum(axis=1)
    df_1[df.columns[4:-1]] = df_1[df.columns[4:-1]].multiply(corr_fact, axis=0)

    # compute the charge and the difference
    charges_of_column = np.array([4, 4, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge = (df_1[df.columns[4:-1]] * charges_of_column).sum(axis=1)
    difference = 12 - charge

    bol = df_1['FeO tot'] < difference
    df_Fe = pd.DataFrame(columns=['Fe2', 'Fe3'], index=df_1.index)

    df_Fe['Fe3'][df_1['FeO tot'] < difference] = df_1['FeO tot']
    df_Fe['Fe2'][df_1['FeO tot'] < difference] = 0

    df_Fe['Fe3'][df_1['FeO tot'] >= difference] = difference
    df_Fe['Fe2'][df_1['FeO tot'] >= difference] = df_1['FeO tot'] - difference
    
    #  corr vr. 1.1
    df_Fe['Fe3'][difference < 0] = 0
    df_Fe['Fe2'][difference < 0] = df_1['FeO tot']
    

    #  We define a new oxide dataframe with different columns for the two Fe and
    #  we calculate the sum over all oxides (to be used in the check)

    df_ox = df.copy()
    df_ox['Fe2O3'] = df_Fe['Fe3'] / corr_fact * mw['Fe2O3'] / 2
    df_ox['FeO'] = df_Fe['Fe2'] / corr_fact * mw['FeO']
    df_ox = pd.concat([df_ox[df_ox.columns[:8]], df_ox[df_ox.columns[-2:]], df_ox[df_ox.columns[9:-2]]], axis=1)
    df_ox['tot'] = df_ox[df_ox.columns[4:-1]].sum(axis=1)

    # We define a new dataframe for cations

    df_cat = pd.concat([df_1[df_1.columns[:8]], df_Fe[['Fe3', 'Fe2']], df_1[df_1.columns[9:]]], axis=1)
    old = df_cat.columns[4:-1]
    new = ['Si', 'Ti', 'Al', 'Cr', 'Fe3', 'Fe2', 'Mn', 'Ni', 'Mg', 'Ca', 'Na', 'K']
    rename_dic = {}
    for i in range(12):
        rename_dic[old[i]] = new[i]
    df_cat = df_cat.rename(columns=rename_dic)
    charges_of_column_new = np.array([4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge_balanced = (df_cat[df_cat.columns[4:-1]] * charges_of_column_new).sum(axis=1)

    # define dataframe for sites

    df_T = pd.DataFrame(columns=['Si', 'Al', 'Ti', 'Fe3'], index=df_cat.index)
    df_M = pd.DataFrame(columns=['Mg', 'Fe2', 'Fe3', 'Al', 'Ti', 'Cr', 'Ni', 'Mn', 'Ca', 'Na', 'K'], index=df_cat.index)

    df_T['Si'] = df_cat['Si']
    df_T['Al'][df_T['Si'] + df_cat['Al'] >= 2] = 2 - df_T['Si']
    df_T['Al'][df_T['Si'] >= 2] = 0
    df_T['Al'][df_T['Si'] + df_cat['Al'] < 2] = df_cat['Al']
    df_T['Ti'][df_T['Si'] + df_T['Al'] + df_cat['Ti'] >= 2] = 2 - (df_T['Si'] + df_T['Al'])
    df_T['Ti'][df_T['Si'] + df_T['Al'] >= 2] = 0
    df_T['Ti'][df_T['Si'] + df_T['Al'] + df_cat['Ti'] < 2] = df_cat['Ti']
    df_T['Fe3'][df_T['Si'] + df_T['Al'] + df_T['Ti'] + df_cat['Ti'] >= 2] = 2 - (df_T['Si'] + df_T['Al'] + df_T['Ti'])
    df_T['Fe3'][df_T['Si'] + df_T['Al'] + df_T['Ti'] + df_cat['Ti'] < 2] = df_cat['Fe3']
    df_T['Fe3'][df_T['Si'] + df_T['Al'] + df_T['Ti'] >= 2] = 0

    df_M = df_cat[df_M.columns]
    df_M[['Al', 'Fe3', 'Ti']] = df_M[['Al', 'Fe3', 'Ti']] - df_T[['Al', 'Fe3', 'Ti']]

    # define dataframe of classification

    df_class = pd.DataFrame(columns=['Fs', 'Wo', 'En', 'Q', 'J'])
    den = df_M[['Fe2', 'Fe3', 'Mn', 'Mg', 'Ca']].sum(axis=1)
    df_class['Fs'] = df_M[['Fe2', 'Fe3', 'Mn']].sum(axis=1) / den * 100
    df_class['Wo'] = df_M['Ca'] / den * 100
    df_class['En'] = df_M['Mg'] / den * 100
    df_class['Q'] = df_M[['Fe2', 'Ca', 'Mg']].sum(axis=1)
    df_class['J'] = df_M[['Na']].sum(axis=1) * 2

    # compute the components and define the relative dataframe
    # we copy the sites dataframes because we have to update them during the computation

    M = df_M.copy()
    M = M.fillna(0)
    T = df_T.copy()

    colcomp = ['CaTiAl2O6', 'CaTs', 'Es', 'CaCrTs', 'NaCrSi2O6', 'Jd', 'Ae', 'Di', 'Hd', 'En(Mg+Ni)', 'Fs(Fe+Mn)']
    df_comp = pd.DataFrame(columns=['Index', 'sample', 'notes', 'notes.1'] + colcomp, index=df.index)

    # computation of first component ('CaTiAl2O6')

    df_comp[colcomp[0]][T[['Al', 'Fe3']].sum(axis=1) >= 2 * M['Ti']] = M['Ti']
    df_comp[colcomp[0]][T[['Al', 'Fe3']].sum(axis=1) < 2 * M['Ti']] = T[['Al', 'Fe3']].sum(axis=1) / 2

    # sites dataframe updating

    T['Si'] = T['Si'] + T['Ti']
    T['Al'] = T['Al'] + T['Fe3'] - 2 * df_comp[colcomp[0]]

    M['Ti'] = M['Ti'] - df_comp[colcomp[0]]
    M['Ca'] = M['Ca'] - df_comp[colcomp[0]]

    # computation of 2nd, 3th, 4th and 5th components ('CaTs','Es','CaCrTs','NaCrSi2O6')

    R1 = M['Al'] * 0
    for i in range(len(R1)):
        if M['Al'].values[i] > 0:
            R1[i] = M['Fe3'].values[i] / M['Al'].values[i]

    df_comp[colcomp[1]][M['Al'] > T['Al'] / (R1 + 1)] = T['Al'] / (R1 + 1)
    df_comp[colcomp[1]][M['Al'] <= T['Al'] / (R1 + 1)] = M['Al']

    df_comp[colcomp[2]][M['Fe3'] > T['Al'] - df_comp[colcomp[1]]] = T['Al'] - df_comp[colcomp[1]]
    df_comp[colcomp[2]][M['Fe3'] <= T['Al'] - df_comp[colcomp[1]]] = M['Fe3']

    df_comp[colcomp[3]][M['Cr'] > T['Al'] - (df_comp[colcomp[2]] + df_comp[colcomp[1]])] = T['Al'] - df_comp[colcomp[2]] - \
                                                                                           df_comp[colcomp[1]]
    df_comp[colcomp[3]][M['Cr'] <= T['Al'] - (df_comp[colcomp[2]] + df_comp[colcomp[1]])] = M['Cr']

    df_comp[colcomp[4]][M['Cr'] - df_comp[colcomp[3]] > M['Na']] = M['Na']
    df_comp[colcomp[4]][M['Cr'] - df_comp[colcomp[3]] <= M['Na']] = M['Cr'] - df_comp[colcomp[3]]

    # sites dataframe updating

    T['Si'] = T['Si'] - (df_comp[colcomp[1]] + df_comp[colcomp[2]] + df_comp[colcomp[3]] + 2 * df_comp[colcomp[4]])
    T['Al'] = T['Al'] - (df_comp[colcomp[1]] + df_comp[colcomp[2]] + df_comp[colcomp[3]])

    M['Fe3'] = M['Fe3'] - df_comp[colcomp[2]]
    M['Al'] = M['Al'] - df_comp[colcomp[1]]
    M['Cr'] = M['Cr'] - df_comp[colcomp[3]] - df_comp[colcomp[4]]
    M['Ca'] = M['Ca'] - df_comp[colcomp[1]] - df_comp[colcomp[2]] - df_comp[colcomp[3]]
    M['Na'] = M['Na'] - df_comp[colcomp[4]]

    # computation of 6th and 7th components ('Jd','Ae')

    df_comp[colcomp[5]][M['Al'] > M['Na'] / (R1 + 1)] = M['Na'] / (R1 + 1)
    df_comp[colcomp[5]][M['Al'] <= M['Na'] / (R1 + 1)] = M['Al']

    df_comp[colcomp[6]][M['Fe3'] > M['Na'] - df_comp[colcomp[5]]] = M['Na'] - df_comp[colcomp[5]]
    df_comp[colcomp[6]][M['Fe3'] <= M['Na'] - df_comp[colcomp[5]]] = M['Fe3']

    # sites dataframe updating

    T['Si'] = T['Si'] - 2 * (df_comp[colcomp[6]] + df_comp[colcomp[5]])

    M['Fe3'] = M['Fe3'] - df_comp[colcomp[6]]
    M['Al'] = M['Al'] - df_comp[colcomp[5]]
    M['Na'] = M['Na'] - (df_comp[colcomp[6]] + df_comp[colcomp[5]])

    # computation of 8th and 9th components ('Di','Hd')

    BOOL1 = pd.DataFrame(columns=['Is_cpx'], index=df.index)
    BOOL1[M['Ca'] > 0] = True
    BOOL1[M['Ca'] <= 0] = False

    R2 = M['Mg'] * 0
    for i in range(len(R2)):
        if M['Mg'].values[i] > 0:
            R2[i] = M['Fe2'].values[i] / M['Mg'].values[i]

    df_comp[colcomp[7]][M['Mg'] > M['Ca'] / (R2 + 1)] = M['Ca'] / (R2 + 1)
    df_comp[colcomp[7]][M['Mg'] <= M['Ca'] / (R2 + 1)] = M['Mg']
    df_comp[colcomp[7]][M['Ca'] <= 0] = 0

    df_comp[colcomp[8]][M['Fe2'] > M['Ca'] - df_comp[colcomp[7]]] = M['Ca'] - df_comp[colcomp[7]]
    df_comp[colcomp[8]][M['Fe2'] <= M['Ca'] - df_comp[colcomp[7]]] = M['Fe2']
    df_comp[colcomp[8]][M['Ca'] <= 0] = 0

    # sites dataframe updating

    T['Si'] = T['Si'] - 2 * (df_comp[colcomp[7]] + df_comp[colcomp[8]])

    M['Mg'] = M['Mg'] - df_comp[colcomp[7]]
    M['Fe2'] = M['Fe2'] - df_comp[colcomp[8]]
    M['Ca'] = M['Ca'] - df_comp[colcomp[8]] - df_comp[colcomp[7]]

    # computation of 10th and 11th components ('En(Mg+Ni)','Fs(Fe+Mn)')

    df_comp[colcomp[9]] = (M['Mg'] + M['Ni']) / 2
    df_comp[colcomp[10]] = (M['Fe2'] + M['Mn']) / 2

    # sites dataframe updating

    T['Si'] = T['Si'] - 2 * (df_comp[colcomp[9]] + df_comp[colcomp[10]])

    Mg_old = M['Mg'].copy()
    Fe2_old = M['Fe2'].copy()

    M['Mg'] = M['Mg'] - (2 * df_comp[colcomp[9]] - M['Ni'])
    M['Fe2'] = M['Fe2'] - (2 * df_comp[colcomp[10]] - M['Mn'])

    M['Ni'] = M['Ni'] - (2 * df_comp[colcomp[9]] - Mg_old)
    M['Mn'] = M['Mn'] - (2 * df_comp[colcomp[10]] - Fe2_old)

    sum_comp = df_comp.sum(axis=1)

    # we define and evaluate the checks

    df_ck = pd.DataFrame(
        columns=['Wo', 'J', 'Fs', 'Wt%', 'components', 'Si apfu', 'CaTiAl2O6', 'T_site', 'M_site', 'charge'],
        index=df.index)

    df_ck['Wo'] = (df_class['Wo'] > 20) & (df_class['Wo'] < 55)
    df_ck['J'] = (df_class['J'] < 1)
    df_ck['Fs'] = (df_class['Fs'] > 5) & (df_class['Fs'] < 50)
    df_ck['Wt%'] = (df_ox['tot'] > 97.5) & (df_ox['tot'] < 102.5)
    df_ck['components'] = (sum_comp > 0.95) & (sum_comp < 1.05)
    df_ck['Si apfu'] = (df_T['Si'] <= 2)
    df_ck['CaTiAl2O6'] = df_comp['CaTiAl2O6'] >= 0
    df_ck['T_site'] = (df_T.sum(axis=1) > 1.95) & (df_T.sum(axis=1) < 2.05)
    df_ck['M_site'] = (df_M.sum(axis=1) > 1.95) & (df_M.sum(axis=1) < 2.05)
    df_ck['charge'] = (charge_balanced > 11.95) & (charge_balanced < 12.05)

    # Global check, if it is False, the element cannot be used in the prediction model
    df_ck['cpx_selection'] = (df_ck == True).all(axis=1)

    # We round the datasets to work with a suitable number of significant digits
    df_comp = df_comp.astype('float64').round(3)
    df_comp[['Index', 'sample', 'notes', 'notes.1']]= df_cat[['Index', 'sample', 'notes', 'notes.1']]
    df_cat = df_cat.round(2)
    df_T = df_T.astype('float64').round(3)
    df_M = df_M.astype('float64').round(3)
    df_class = df_class.round(2)

    # the output dictionary is defined
    output_dictionary = {'components': df_comp, 'cations': df_cat, 'checks': df_ck, 'classifications': df_class,
                         'site_T': df_T, 'site_M1&2': df_M, 'sum_of_components': sum_comp}

    return output_dictionary
