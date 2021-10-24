"""
This example shows how to define your own dataset and use it to train a
non-trivial GNN with message-passing and pooling layers.
The script also shows how to implement fast training and evaluation functions
in disjoint mode, with early stopping and accuracy monitoring.
The dataset that we create is a simple synthetic task in which we have random
graphs with randomly-colored nodes. The goal is to classify each graph with the
color that occurs the most on its nodes. For example, given a graph with 2
colors and 3 nodes:
x = [[1, 0],
     [1, 0],
     [0, 1]],
the corresponding target will be [1, 0].
"""
# https://arxiv.org/pdf/2007.13681.pdf

# 69


# 57

print('\n\n\n')
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool, GlobalAttentionPool
from spektral.transforms.normalize_adj import NormalizeAdj

from spektral.layers import GCNConv, GlobalSumPool

from generator import generate_VELO_dataset

import matplotlib.pyplot as plt

class VELO_dataset(Dataset):

    def __init__(self, n_samples, n_min=10, n_max=25, **kwargs):
        self.n_samples = n_samples
        self.n_min = n_min
        self.n_max = n_max
        super().__init__(**kwargs)

    def read(self):
        def make_graph():
            # n = np.random.randint(self.n_min, self.n_max)
            n = 10 
            x, y, _bin = generate_VELO_dataset(n)

            a = np.ones((np.shape(y)[0],np.shape(y)[0])).astype('int')
            a = sp.csr_matrix(a)

            return Graph(x=x, a=a, y=y)

        return [make_graph() for _ in range(self.n_samples)]


dataset = VELO_dataset(25000, transforms=NormalizeAdj())


class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv1 = GCSConv(n_hidden)
        self.graph_conv2 = GCSConv(n_hidden)
        # self.graph_conv3 = GCSConv(int(n_hidden/4.))
        self.leaky = LeakyReLU(alpha=0.2)
        self.dropout = Dropout(0.25)
        # self.pool1 = GlobalAttentionPool(3)

        # self.pool1 = TopKPool(ratio=0.5)
        # self.graph_conv2 = GCNConv(n_hidden)
        # self.dense = Dense(n_labels, 'sigmoid')
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        x, a = inputs
        H = self.graph_conv1([x,a])
        # H, a = self.pool1(H)
        # H = self.graph_conv2([H,a])
        H = self.leaky(H)
        # H = self.dropout(H)
        # H = self.graph_conv2([H,a])
        # H = self.leaky(H)
        # H = self.dropout(H)
        # H = self.graph_conv3([H,a])
        # H = self.leaky(H)
        # H = self.dropout(H)
        # x1, a1, i1 = self.pool1([x, a, i])
        # H = self.graph_conv2(H)
        out = self.dense(H)

        return out



# optimizer = Adam(lr=1e-2, beta_1=0.5, decay=0, amsgrad=True)
optimizer = Adam(lr=1e-2)

loss_fn = SparseCategoricalCrossentropy()
model = MyFirstGNN(32, 2) # 2 categories if SparseCategoricalCrossentropy

# loss_fn = MeanSquaredError()
# model = MyFirstGNN(256, 1) # 2 categories if SparseCategoricalCrossentropy

model.compile()



# from spektral.data import DisjointLoader
# loader = DisjointLoader(dataset, batch_size=3)
from spektral.data import BatchLoader
loader = BatchLoader(dataset, batch_size=10)



@tf.function()
def train(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        x, a = inputs
        predictions = model([x,a], training=True)
        target = tf.squeeze(target)
        predictions = tf.squeeze(predictions)
        loss = loss_fn(target, predictions) + sum(model.losses)
    A = model.trainable_variables
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    B = model.trainable_variables
    return loss, predictions, target

@tf.function()
def query(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        x, a = inputs
        predictions = model([x,a], training=False)
        predictions = tf.squeeze(predictions)
    return predictions, target



class VELO_dataset_plotting(Dataset):

    def __init__(self, n_samples, n_min=10, n_max=25, **kwargs):
        self.n_samples = n_samples
        self.n_min = n_min
        self.n_max = n_max
        super().__init__(**kwargs)

    def read(self):
        def make_graph():
            # n = np.random.randint(self.n_min, self.n_max)
            n = 10 
            x, y, plotting = generate_VELO_dataset(n)

            a = np.ones((np.shape(y)[0],np.shape(y)[0])).astype('int')
            a = sp.csr_matrix(a)

            return Graph(x=x, a=a, y=plotting)
            # return Graph(x=x, a=a, y=y)

        return [make_graph() for _ in range(self.n_samples)]









losses = np.empty(0)
for idx, batch in enumerate(loader):

    A, B, C = train(*batch)
    losses = np.append(losses,A.numpy())

    if idx % 10000 == 0:

        print('\n',idx,'loss:',A.numpy())

        print(B.numpy()[0][:,1],C.numpy()[0])

        if idx == 0: continue

        plt.plot(losses)
        plt.savefig('loss')
        plt.close('all')

        print(' ')
        print('Plot a test...')
        nTests = 3
        plt.figure(figsize=(8,nTests*4))

        dataset_plotting = VELO_dataset_plotting(nTests, transforms=NormalizeAdj())
        loader_plotting = BatchLoader(dataset_plotting, batch_size=1)

        for jdx, plotBatch in enumerate(loader_plotting):

            if jdx == nTests:
                break

            predictions, plotting = query(*plotBatch)

            predictions = predictions.numpy()[:,1]
            plotting = plotting.numpy()[0]


            points = plotting[0]
            points_at_layers = plotting[1]

            plt.subplot(nTests,2,jdx*2+1)
            plt.title('Truth')
            for kdx in range(np.shape(points)[0]):   
                if points[kdx][2][0] < 1.:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:blue',alpha=0.25)
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=0.25,color='k', marker='x')
                else:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:red')
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=1.,color='k', marker='x')

            plt.xlim(-1.5,10)
            plt.ylim(-3,3)
            plt.axhline(y=0,c='k',alpha=0.1)
            plt.axvline(x=0, c='k',alpha=0.1)
            plt.axvline(x=7, c='k',alpha=1)
            plt.axvline(x=9, c='k',alpha=1)

            plt.subplot(nTests,2,jdx*2+2)
            plt.title('GNN')
            for kdx in range(np.shape(points)[0]):   
                if predictions[kdx] < 0.9:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:blue',alpha=0.25)
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=0.25,color='k', marker='x')
                else:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:red')
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=1.,color='k', marker='x')

            plt.xlim(-1.5,10)
            plt.ylim(-3,3)
            plt.axhline(y=0,c='k',alpha=0.1)
            plt.axvline(x=0, c='k',alpha=0.1)
            plt.axvline(x=7, c='k',alpha=1)
            plt.axvline(x=9, c='k',alpha=1)

        plt.savefig('examples_%d'%idx,bbox_inches='tight')
        plt.close('all')
        quit()
