# Use this code to train an IC50 predictor using labeled cell-line screening data
from keras import models
import pickle
from keras.layers import Dense
from keras.layers import Merge
from keras.callbacks import EarlyStopping
import numpy as np

# load tabular data
def load_data(filename, log_trans=False, label=False):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    if label:
        data_labels = lines[1].replace('\n', '').split('\t')[1:]
        dx = 2
    else:
        dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    if log_trans:
        data= np.log2(data + 1)
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names

if __name__ == '__main__':
    # load tabular input mutation and expression datasets, and output IC50 data
    # datasets should be prepared in a dara frame-like format (with row/column labels) and saved as txt files
    # columns represent samples (cell lines) and rows are features (genes or drugs)
    # please refer to our paper for details regarding data preparation/filtering and sources
    data_mut, data_labels_mut, sample_names_mut, gene_names_mut = load_data("/path/to/your/ccle_mut_data.txt")
    data_exp, data_labels_exp, sample_names_exp, gene_names_exp = load_data("/path/to/your/ccle_exp_data.txt")
    data_drug, data_labels_drug, sample_names_drug, drug_names_drug = load_data("/path/to/your/ccle_ic50_data.txt")

    # load parameters of autoencoders pretrained on unlabeled tumor mutation/expression data
    # use Pretrain.py to generate the required pickle files
    premodel_mut = pickle.load(open('tcga_pretrained_autoencoder_mut.pickle', 'rb'))
    premodel_exp = pickle.load(open('tcga_pretrained_autoencoder_exp.pickle', 'rb'))

    # set hyperparameters
    activation_func = 'relu'
    activation_func2 = 'linear'
    init = 'he_uniform'
    dense_layer_dim = 128
    batch_size = 16
    num_epoch = 50

    # divide samples into 90% (training/validation) and 10% (testing)
    id_rand = np.random.permutation(data_drug.shape[0])
    id_train = id_rand[0:round(data_drug.shape[0]*0.9)]
    id_test = id_rand[round(data_drug.shape[0]*0.9):data_drug.shape[0]]

    # model construction and training/testing
    model_mut = models.Sequential()
    model_mut.add(Dense(output_dim=1024, input_dim=premodel_mut[0][0].shape[0], activation=activation_func, weights=premodel_mut[0]))
    model_mut.add(Dense(output_dim=256, input_dim=1024, activation=activation_func, weights=premodel_mut[1]))
    model_mut.add(Dense(output_dim=64, input_dim=256, activation=activation_func, weights=premodel_mut[2]))

    model_exp = models.Sequential()
    model_exp.add(Dense(output_dim=1024, input_dim=premodel_exp[0][0].shape[0], activation=activation_func, weights=premodel_exp[0]))
    model_exp.add(Dense(output_dim=256, input_dim=1024, activation=activation_func, weights=premodel_exp[1]))
    model_exp.add(Dense(output_dim=64, input_dim=256, activation=activation_func, weights=premodel_exp[2]))

    model_final = models.Sequential()
    model_final.add(Merge([model_mut, model_exp], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=128, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=data_drug.shape[1], input_dim=dense_layer_dim, activation=activation_func2, init=init))

    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit([data_mut[id_train], data_exp[id_train]], data_drug[id_train], nb_epoch=num_epoch,
                    validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[history])
    cost_testing = model_final.evaluate([data_mut[id_test], data_exp[id_test]], data_drug[id_test], verbose=0, batch_size=batch_size)
    print('Training completed.\nTesing cost=%.4f' % cost_testing)

    # save trained model
    model_final.save("model_final.h5")

    # predict drug response using a trained model and save outputs
    data_pred = model_final.predict([data_mut, data_exp], batch_size=batch_size, verbose=0)
    # change input data to the samples you would like to predict (unscreened cell lines, tumors, etc.)
    np.savetxt('predicted_IC50.txt', np.transpose(data_pred), delimiter='\t', fmt='%.4f')
