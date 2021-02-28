# Use this code to pretrain an autoencoder using unlabeled gene mutation or gene expression data of tumors
# Output of this code is used by TrainModel.py to initialize the weights of IC50 predictor on cell lines
from keras import models
import pickle
import numpy as np
from keras.layers import Dense

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

# save model parameters to pickle
def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

if __name__ == '__main__':
    # load tabular mutation or expression data
    # dataset should be prepared in a dara frame-like format (with row/column labels) and saved as txt file
    # columns represent samples (tumors) and rows are features (genes)
    # please refer to our paper for details regarding data preparation/filtering and sources
    data, data_labels, sample_names, gene_names = load_data("/path/to/your/tcga_mut_data.txt")
    input_dim = data.shape[1]

    # set hyperparameters
    first_layer_dim = 1024
    second_layer_dim = 256
    third_layer_dim = 64
    batch_size = 64
    epoch_size = 100
    activation_func = 'relu'
    init = 'he_uniform'

    # model construction and training
    model = models.Sequential()
    model.add(Dense(output_dim=first_layer_dim, input_dim=input_dim, activation=activation_func, init=init))
    model.add(Dense(output_dim=second_layer_dim, input_dim=first_layer_dim, activation=activation_func, init=init))
    model.add(Dense(output_dim=third_layer_dim, input_dim=second_layer_dim, activation=activation_func, init=init))
    model.add(Dense(output_dim=second_layer_dim, input_dim=third_layer_dim, activation=activation_func, init=init))
    model.add(Dense(output_dim=first_layer_dim, input_dim=second_layer_dim, activation=activation_func, init=init))
    model.add(Dense(output_dim=input_dim, input_dim=first_layer_dim, activation=activation_func, init=init))

    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit(data, data, nb_epoch=epoch_size, batch_size=batch_size, shuffle=True)

    cost = model.evaluate(data, data, verbose=0)
    print('Training completed.\nCost=%.4f' % cost)

    # save model parameters to pickle file, which will be used in TrainModel.py
    save_weight_to_pickle(model, 'tcga_pretrained_autoencoder_mut.pickle')
