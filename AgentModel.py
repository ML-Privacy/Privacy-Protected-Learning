from keras.models import Model
import tensorflow as tf
import numpy as np


class AgentModel(Model):
    def __init__(self, inputs, outputs, nb_agents, sparsity=False, identical=True):
        super().__init__(inputs=inputs, outputs=outputs)
        self.nb_agents = nb_agents

        if sparsity and nb_agents > 3:
            self.pi = np.zeros((nb_agents, nb_agents))
            for i in range(nb_agents):
                self.pi[i][(i + 1) % nb_agents] = 0.33
                self.pi[i][(i + nb_agents - 1) % nb_agents] = 0.33
                self.pi[i][i] = 0.34
        else:
            self.pi = np.ones((nb_agents, nb_agents)) / nb_agents

        self.varLength = len(self.trainable_variables)

        if identical:
            self.format_weights()


    def format_weights(self):
        i=0
        while (i < len(self.layers)):
            j = i
            if self.layers[i].get_weights() != []:
                while (j - i < self.nb_agents - 1):
                    # print(j)
                    j += 1
                    self.layers[j].set_weights(self.layers[i].get_weights())
            i += self.nb_agents

    def train_step2(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables


        trainable_vars = trainable_vars[0:self.varLength]
        gradients = tape.gradient(loss, trainable_vars)

        #self.consensus(trainable_vars)
        
        self.optimizer.trainStep(gradients, trainable_vars)
        #self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def consensus(self, trainVars):
        
        layers = int(len(trainVars) / (2 * self.nb_agents))

        conVar = [None] * len(trainVars)

        for k in range(layers):
            for i in range(self.nb_agents):
                newVar = trainVars[2*(i + self.nb_agents * k)] * 0
                newVarb = trainVars[2*(i + self.nb_agents * k) + 1] * 0

                for j in range(self.nb_agents):
                    newVar = newVar + self.pi[i][j] * trainVars[2*(j + self.nb_agents * k)]
                    newVarb = newVarb + self.pi[i][j] * trainVars[2*(j + self.nb_agents * k) + 1]
                conVar[2*(i + self.nb_agents * k)] = newVar
                conVar[2*(i + self.nb_agents * k)+1] = newVarb
        

        for i in range(len(trainVars)):
            trainVars[i].assign(conVar[i])

    def consensus2(self, trainVars):
        
        layers = int(len(trainVars) / (2 * self.nb_agents))

        if self.conVars == None:

            self.conVars = [None] * len(trainVars)

            for i in range(len(trainVars)):
                self.conVars[i] = tf.Variable(trainVars[i].initialized_value())

        

        if self.consense:
            for i in range(len(trainVars)):
                self.conVars[i].assign(trainVars[i])
            self.consense = False

        conVar = [None] * len(trainVars)

        for k in range(layers):
            for i in range(self.nb_agents):
                newVar = trainVars[2*(i + self.nb_agents * k)] * 0
                newVarb = trainVars[2*(i + self.nb_agents * k) + 1] * 0

                for j in range(self.nb_agents):
                    if i == j:
                        newVar = newVar + self.pi[i][j] * trainVars[2*(j + self.nb_agents * k)]
                        newVarb = newVarb + self.pi[i][j] * trainVars[2*(j + self.nb_agents * k) + 1]
                    else:
                        newVar = newVar + self.pi[i][j] * self.conVars[2*(j + self.nb_agents * k)]
                        newVarb = newVarb + self.pi[i][j] * self.conVars[2*(j + self.nb_agents * k) + 1]
                conVar[2*(i + self.nb_agents * k)] = newVar
                conVar[2*(i + self.nb_agents * k)+1] = newVarb
        

        for i in range(len(trainVars)):
            trainVars[i].assign(conVar[i])

                










