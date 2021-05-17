import os
import json
import Models
import numpy as np
import csv
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import np_utils, generic_utils
from keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import Adam, SGD
#from CDSGD import CDSGD, model_compilers_cdsgd, update_parameters_cdsgd
from keras.models import model_from_json
import time

from CDGD import CDGD
from QCDGD import QCDGD
from Params import Params

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='set data type as mnist or cifar10')
parser.add_argument('--type', help='Set the type of training as base or new')
args = parser.parse_args()

def _make_batches(size, batch_size):
    num_batches = int(np.floor(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def _slice_arrays(arrays, start=None, stop=None):
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]

def decayed_learning_rate(step):
  step = min(step, decay_steps)
  return ((initial_learning_rate - end_learning_rate) *
          (1 - step / decay_steps)**(power)
         ) + end_learning_rate

def train(model_name="CNN",
          batch_size = 32,
          nb_epoch = 2000,
          dataset = "mnist",
          optimizer = "CDGD",
          nb_agents=5,
          step_eval=20,
		  **kwargs):

    
    paramExpla = ["model_name", "optimizer", "dataset", "nb_epoch", "batch_size", "nb_agents"] + list(kwargs.keys())
    parameters = [model_name, optimizer, dataset, nb_epoch, batch_size, nb_agents] + list(kwargs.values())

    print('\nStarting Process:')
    print(list(zip(paramExpla,parameters)))
    
    if dataset == "cifar10":
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    if dataset == "cifar100":
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
    if dataset == "mnist":
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    img_dim = X_train.shape[-3:]
    nb_classes = len(np.unique(Y_train))

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    ins=[X_train,Y_train]
    num_train_samples = ins[0].shape[0]
    agent_data_size= (num_train_samples//nb_agents)


    x_data = {}
    y_data = {}
    x_vali = {}
    y_vali = {}

    for i in range(nb_agents):
        x_data['input_'+str(i+1)] = X_train[i*agent_data_size:(i+1)*agent_data_size]
        y_data['d'+str(i+1)] = Y_train[i*agent_data_size:(i+1)*agent_data_size]
        x_vali['input_'+str(i+1)] = X_test
        y_vali['d'+str(i+1)] = Y_test

    test_batch_size = 1024
    if 'test_batch_size' in kwargs:
        test_batch_size = kwargs['test_batch_size']

    trainData = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(1024).repeat().batch(batch_size).prefetch(1)
    testData  = tf.data.Dataset.from_tensor_slices((x_vali, y_vali)).batch(test_batch_size)
    
    #print(trainData)
    #print(testData)


    lr = 1e-3
    
    if 'lr' in kwargs:
        lr = kwargs['lr']
        
        if lr == 'PolynomialDecay':
            
            if not 'starter_learning_rate' in kwargs:
                raise ValueError('For TernGrad, must specify the "starter_learning_rate" parameter')
            
            if not 'decay_steps' in kwargs:
                raise ValueError('For TernGrad, must specify the "decay_steps" parameter')
            
            if not 'end_learning_rate' in kwargs:
                raise ValueError('For TernGrad, must specify the "end_learning_rate" parameter')
            
            if not 'power' in kwargs:
                raise ValueError('For TernGrad, must specify the "power" parameter')
            
            lr = tf.keras.optimizers.schedules.PolynomialDecay(kwargs['starter_learning_rate'],
                                                            kwargs['decay_steps'],
                                                            kwargs['end_learning_rate'],
                                                            power=kwargs['power'])
                        

    if nb_agents != 1:
        
        
        topology = 'full'
        if 'topology' in kwargs:
            topology = kwargs['topology']
        
        always_update = False
        if 'always_update' in kwargs:
            always_update = kwargs['always_update']


        big_k=1
        maxLam=0.01

        params = Params(nb_agents, big_k, always_update=always_update, topology=topology)
    

    if optimizer == "CDGD":
        
        if not 'c1' in kwargs:
            raise ValueError('For CDGD, must specify the "c1" parameter')
        
        if not 'delta' in kwargs:
            raise ValueError('For CDGD, must specify the "delta" parameter')
        
        params = Params(nb_agents, big_k, always_update=always_update, topology=topology, maxLam=maxLam)
        opt = CDGD(lr=1E-2, decay=0, nesterov=False, nb_agents=nb_agents, params=params, c1=kwargs['c1'], delta=kwargs['delta'])


    elif optimizer == "QCDGD":
        
        if not 'c1' in kwargs:
            raise ValueError('For QDGD, must specify the "c1" parameter')
        
        if not 'clipStd' in kwargs:
            raise ValueError('For QDGD, must specify the "clipStd" parameter')
        
        if not 'ternSt' in kwargs:
            raise ValueError('For QDGD, must specify the "ternSt" parameter')

        if nb_agents == 1:
            big_k=1
            maxLam=0.01
            topology = 'full'
            always_update = False
        
        params = Params(nb_agents, big_k, always_update=always_update, topology=topology, maxLam=maxLam)
        opt = QCDGD(lr=1E-2, decay=0, nesterov=False, nb_agents=nb_agents, params=params, ternSt=kwargs['ternSt'], clip=kwargs['clipStd'], c1=kwargs['c1'])
    
    initer = 'glorot_uniform'
    if 'initer' in kwargs:
        initer = kwargs['initer']

    identical = True
    if 'identical' in kwargs:
        identical = kwargs['identical']

    model = Models.load(model_name, img_dim, nb_classes, opt, nb_agents=nb_agents, identical=identical, kernel_initializer=initer)
        
    # model.summary()
        #stop
    
    step_list = []
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    
    #data = model.fit(X_agent_ins, Y_agent_ins, validation_data=(x_validation, y_validation), epochs=nb_epoch // step_eval, steps_per_epoch=step_eval, batch_size=batch_size, shuffle=True)
    data = model.fit(trainData, validation_data=testData, epochs=nb_epoch // step_eval, steps_per_epoch=step_eval, batch_size=batch_size, shuffle=True)
    #data = model.fit(generator, validation_data=(x_validation, y_validation), epochs=nb_epoch // step_eval, steps_per_epoch=step_eval, batch_size=batch_size, shuffle=True)

    keys = list(data.history.keys())
    
    offset = 1
    if nb_agents == 1:
        offset = 0

    train_losses = [ sum(x) for x in zip(*[data.history.get(key) for key in keys[offset:offset+nb_agents]]) ]
    train_accs = [ sum(x) / nb_agents for x in zip(*[data.history.get(key) for key in keys[offset+nb_agents:offset+2*nb_agents]]) ]
    val_losses = [ sum(x) for x in zip(*[data.history.get(key) for key in keys[2*offset+2*nb_agents:2*offset+3*nb_agents]]) ]
    val_accs = [ sum(x) / nb_agents for x in zip(*[data.history.get(key) for key in keys[2*offset+3*nb_agents:2*offset+4*nb_agents]]) ]
    
    step_list = [*range(step_eval, nb_epoch + 1, step_eval)]
    del model

    return paramExpla, parameters, step_list, train_losses, train_accs, val_losses, val_accs


def save(file, paramExp, parameters, step_list, train_losses, train_accs, val_losses, val_accs):

    file.writerow(['Experiment'])
    file.writerow(paramExp)
    file.writerow(parameters)
    file.writerow(['Epoch Iterations'])
    file.writerow(step_list)
    file.writerow(['train_losses'])
    file.writerow(train_losses)
    file.writerow(['train_accs'])
    file.writerow(train_accs)
    file.writerow(['val_losses'])
    file.writerow(val_losses)
    file.writerow(['val_accs'])
    file.writerow(val_accs)
    file.writerow([])

supported_datasets = ["mnist", "cifar10", "cifar100"]
supported_optimizers = ["CDGD", "QCDGD"]
optimizer_parameters = {"CDGD":{"c1": "float", "delta": "float"},
                        "QCDGD":{"clipStd": "float", "ternSt": "float", "c1": "float"}}

if args.data:
	global data_type
	data_type = args.data

if args.type:
	test_type = args.type
else:
	raise ValueError('Must specify the type as base or new')

if __name__ == '__main__':
    
    if data_type == 'mnist':
        
        filename = 'mnist_training_'+ test_type + '.csv'

        if test_type == 'base':

            rData = train(model_name="CNN", batch_size = 32, 
                    nb_epoch = 2000, dataset = "mnist", 
                    optimizer = "CDGD",
                    nb_agents=4, step_eval=50,
                    c1=.007, delta=3/2, clipStd=2.5, ternSt=1)
        else:

            rData = train(model_name="CNN", batch_size = 32, 
                    nb_epoch = 2000, dataset = "mnist", 
                    optimizer = "QCDGD",
                    nb_agents=4, step_eval=50,
                    c1=.007, delta=3/2, clipStd=2.5, ternSt=1)

        
    elif data_type == 'cifar10':

        filename = 'cifar10_training_'+ test_type + '.csv'

        if test_type == 'base':

            rData = train(model_name="res", batch_size = 32, 
                    nb_epoch = 50000, dataset = "cifar10", 
                    optimizer = "CDGD",
                    nb_agents=4, step_eval=250,
                    c1=.007, delta=3/2, clipStd=2.5, ternSt=1)

        else:

            rData = train(model_name="res", batch_size = 32, 
                    nb_epoch = 50000, dataset = "cifar10", 
                    optimizer = "QCDGD",
                    nb_agents=4, step_eval=250,
                    c1=.007, delta=3/2, clipStd=2.5, ternSt=1)
            


    with open(filename, mode='a') as csv_file:
        file = csv.writer(csv_file, lineterminator = '\n')
        save(file, *rData)
    tf.keras.backend.clear_session()

