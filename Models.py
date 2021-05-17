from __future__ import absolute_import
from __future__ import division

import tensorflow
from tensorflow import Tensor
from keras.models import Model
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, ZeroPadding2D, Activation, Add, AveragePooling2D, GlobalAveragePooling2D
from keras import activations, regularizers
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense # Toggle off
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from keras.layers.merge import concatenate
# from tensorflow.keras.applications.resnet50 import ResNet50
# from ResNet import ResNet20ForCIFAR10, ResNet32ForCIFAR10

# For resshoji model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add, Input, Flatten
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

from AgentModel import AgentModel
# from Official_resnet import cifar10_resnet_v2_generator
# from resnetModel import resnet_18, resnet_50, resnet_101
# from residual_block import make_basic_block_layer, make_bottleneck_layer


def standard_conv_block(x, nb_filter, kernel_initializer, bias_initializer, subsample=(1,1), pooling=False, bn=False, dropout_rate=None, name=None, activation='relu'):
    x = Conv2D(nb_filter, (3, 3), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, padding="same", name = name)(x)
    if bn:
        x = BatchNormalization(mode=2, axis=1)(x)
    if activation == 'relu':
        x = Activation(activation)(x)
        if pooling:
            x = MaxPooling2D()(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x
    else:
        x = Activation(activation)(x)
        if pooling:
            x = AveragePooling2D()(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x


def CNN(img_dim, nb_classes, kernel_initializer, bias_initializer, activation='relu', outName='NetOut'):

    x_input = Input(shape=img_dim)

    x = standard_conv_block(x_input, 32, kernel_initializer, bias_initializer, activation=activation)
    x = standard_conv_block(x, 32, kernel_initializer, bias_initializer, pooling=True, activation=activation)
    x = standard_conv_block(x, 64, kernel_initializer, bias_initializer, activation=activation)
    x = standard_conv_block(x, 64, kernel_initializer, bias_initializer, pooling=True, dropout_rate=0.25, activation=activation)

    # FC part
    x = Flatten()(x)
    x = Dense(512, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Dense(nb_classes, activation="softmax", name=outName, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)

    return x_input, x

def res(img_dim, nb_classes, a, b, outName):
    def res_net_block(input_data, filters, conv_size):
        x = Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = BatchNormalization()(x)
        x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, input_data])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=img_dim)
    x = Conv2D(32, 3, activation='relu')(inputs)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D(3)(x)
    num_res_net_blocks = 8
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    
    x = Conv2D(64, 3, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(10, activation='softmax', name=outName)(x)
    return inputs, outputs


def agentCNN(func, img_dim, nb_classes, nb_agents, kernel_initializer, bias_initializer, model_name):

    x_inputs = [None] * nb_agents
    x_outputs = [None] * nb_agents

    names = [None] * nb_agents
    
    for i in range(nb_agents):
        x_inputs[i], x_outputs[i] = func(img_dim, nb_classes, kernel_initializer, bias_initializer, outName='d'+str(i+1))
    # mod1 = Model(inputs = x_inputs[0], outputs = x_outputs[0])
    # mod1 = Model(inputs = x_inputs, outputs = x_outputs)
    # mod1.summary()

    return x_inputs, x_outputs



supported_models = ["CNN", "Big_CNN", "FCN", "LeNet5","res"]

def load(model_name, img_dim, nb_classes, opt, loss='categorical_crossentropy', nb_agents=5, sparsity=False, identical=True, kernel_initializer='lecun_uniform', bias_initializer='zeros', activation=None):
    
    func = None

    if model_name == "CNN":
        func = CNN
    elif model_name == "res":
        func = res

    if nb_agents == 1 and False:
        x, y = func(img_dim, nb_classes, kernel_initializer, bias_initializer, activation=activation)
        model = Model(inputs=[x], outputs=[y])
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    else:
        x, y = agentCNN(func, img_dim, nb_classes, nb_agents, kernel_initializer, bias_initializer, model_name)
        model = AgentModel(x, y, nb_agents, sparsity=sparsity, identical=identical)
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        # model.summary()
        # print(len(model.layers))
        # model.format_weights()

    return model
