'''
    
    Simple example of node classification using a simple toy model of a displaced vertex.

'''
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv
from spektral.transforms.normalize_adj import NormalizeAdj

from generator import generate_dataset

class getDataset(Dataset):

    def __init__(self, n_samples, plotting=False, n_min=10, n_max=25, **kwargs):
        self.n_samples = n_samples
        self.n_min = n_min
        self.n_max = n_max
        self.plotting = plotting
        super().__init__(**kwargs)

    def read(self):
        def make_graph():
            n = np.random.randint(self.n_min, self.n_max)

            x, y, _bin = generate_dataset(n)

            a = np.ones((np.shape(y)[0],np.shape(y)[0])).astype('int')
            a = sp.csr_matrix(a)

            if self.plotting:
                return Graph(x=x, a=a, y=_bin)
            else:
                return Graph(x=x, a=a, y=y)

        return [make_graph() for _ in range(self.n_samples)]

class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv1 = GCSConv(n_hidden)
        self.graph_conv2 = GCSConv(n_hidden)
        self.graph_conv3 = GCSConv(n_hidden)
        self.leaky = LeakyReLU(alpha=0.2)
        self.dense = Dense(n_labels, 'tanh')

    def call(self, inputs):
        x, a, i = inputs
        H = self.graph_conv1([x,a])
        H = self.leaky(H)
        H = self.graph_conv2([H,a])
        H = self.leaky(H)
        H = self.graph_conv3([H,a])
        H = self.leaky(H)
        out = self.dense(H)

        return out

@tf.function(experimental_relax_shapes=True)
def train(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions, target

@tf.function(experimental_relax_shapes=True)
def query(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        x, a, i = inputs
        predictions = model([x,a,i], training=False)
    return predictions, target


# Create the network
optimizer = Adam(lr=1e-2)
# loss_fn = SparseCategoricalCrossentropy()
loss_fn = MeanSquaredError()
model = MyFirstGNN(32, 2) # 2 categories if SparseCategoricalCrossentropy
model.compile()

# Create the dataset
dataset = getDataset(25000, transforms=NormalizeAdj())
loader = DisjointLoader(dataset, batch_size=50, node_level=True)


save_idx = 5000
losses = np.empty(0)

# Enter training loop
for idx, batch in enumerate(loader):

    A, B, C = train(*batch)
    losses = np.append(losses,A.numpy())

    if idx % 500 == 0: print(idx)

    # Saving...
    if idx % save_idx == 0:

        print(B, C)
        # continue

        if idx == 0: continue

        print('Saving...')
        print(idx,'loss:',A.numpy())

        plt.plot(losses)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.savefig('loss')
        plt.close('all')

        print(' ')
        print('Plot a test...')
        nTests = 6
        plt.figure(figsize=(16,nTests*4))

        dataset_plotting = getDataset(nTests, plotting=True, transforms=NormalizeAdj())
        loader_plotting = DisjointLoader(dataset_plotting, batch_size=1)

        for jdx, plotBatch in enumerate(loader_plotting):

            if jdx == nTests:
                break

            predictions, plotting = query(*plotBatch)

            predictions = predictions.numpy()

            predictions[:,0] = predictions[:,0]*6.
            predictions[:,1] = (predictions[:,1] - 0.5)*4.

            plotting = plotting.numpy()[0]

            points = plotting[0]
            points_at_layers = plotting[1]

            nTracks = [0, 0, 0]
            ax = plt.subplot(nTests,4,jdx*2+1)
            plt.title('Truth')
            for kdx in range(np.shape(points)[0]):   
                nTracks[0] += 1
                if points[kdx][2][0] < 1.:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:blue',alpha=0.25)
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=0.25,color='k', marker='x')
                else:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:red')
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=1.,color='k', marker='x')
                    nTracks[1] += 1

            plt.xlim(-1.5,10)
            plt.ylim(-3,3)
            plt.axhline(y=0,c='k',alpha=0.1)
            plt.axvline(x=0, c='k',alpha=0.1)
            plt.axvline(x=7, c='k',alpha=1)
            plt.axvline(x=9, c='k',alpha=1)

            ax = plt.subplot(nTests,4,jdx*2+2)
            plt.title('GNN',color='tab:orange')
            for kdx in range(np.shape(points)[0]):   
                if predictions[kdx][0] < 0.1 and predictions[kdx][1] < 0.1:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:blue',alpha=0.25)
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=0.25,color='k', marker='x')
                else:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:red')
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=1.,color='k', marker='x')
                    plt.scatter(predictions[kdx][0],predictions[kdx][1],alpha=1.,color='tab:red', marker='x',s=50)
                    nTracks[2] += 1

            plt.text(0.05, 0.15, 'nTracks: %d'%nTracks[0], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0.05, 0.1, 'nTracks signal (truth): %d'%nTracks[1], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0.05, 0.05, 'nTracks signal (GNN): %d'%nTracks[2], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

            plt.xlim(-1.5,10)
            plt.ylim(-3,3)
            plt.axhline(y=0,c='k',alpha=0.1)
            plt.axvline(x=0, c='k',alpha=0.1)
            plt.axvline(x=7, c='k',alpha=1)
            plt.axvline(x=9, c='k',alpha=1)

        plt.savefig('tests_%d'%idx,bbox_inches='tight')
        plt.close('all')