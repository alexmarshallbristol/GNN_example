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
from spektral.layers import GlobalAvgPool
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

class GNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv1 = GCSConv(n_hidden)
        self.graph_conv2 = GCSConv(n_hidden)
        self.graph_conv3 = GCSConv(n_hidden)
        self.leaky = LeakyReLU(alpha=0.2)
        self.dense = Dense(n_labels, 'softmax')
        self.dense2 = Dense(2, 'tanh')
        self.pool = GlobalAvgPool()

    def call(self, inputs):
        x, a, i = inputs
        H = self.graph_conv1([x,a])
        H = self.leaky(H)

        H = self.graph_conv2([H,a])
        H = self.leaky(H)

        H = self.graph_conv3([H,a])
        H = self.leaky(H)

        classes = self.dense(H)

        H = self.pool([x,i])
        origin = self.dense2(H)
        
        return classes, origin

@tf.function(experimental_relax_shapes=True)
def train(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        x, a, i = inputs
        predictions, predictions_origin = model(inputs,training=True)

        classes_target = target[:,2]
        origin_target = target[:,:2]

        # Bodge job of obtaining labels for the origin vertex. Unclear how to save mulitple different shaped label ojects in spektral - so this is a real fudge.
        i += 1
        i_plus = i - tf.roll(i, shift=-1, axis=0)
        i_plus = tf.cast(i_plus,'float64')
        origin_target = tf.squeeze(origin_target)
        i_plus = tf.squeeze(i_plus)
        i_plus = tf.expand_dims(i_plus,axis=1)
        origin_target = tf.concat([origin_target,i_plus],axis=1)
        origin_target = tf.gather(origin_target, tf.where((origin_target[:,2]!=0))[:, 0])[:,:2]

        loss = loss_scc(classes_target, predictions) + 10.*loss_mse(origin_target, predictions_origin) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def query(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        x, a, i = inputs
        predictions, predictions_origin = model([x,a,i], training=False)
    return predictions, predictions_origin, target


# Create the network
optimizer = Adam(lr=1e-2, beta_1=0.5, decay=0, amsgrad=True)
loss_scc = SparseCategoricalCrossentropy()
loss_mse = MeanSquaredError()
model = GNN(32, 2) # 2 categories if SparseCategoricalCrossentropy
model.compile()

# Create the dataset
dataset = getDataset(25000, transforms=NormalizeAdj())
loader = DisjointLoader(dataset, batch_size=50, node_level=True)


save_idx = 5000
losses = np.empty(0)

# Enter training loop
for idx, batch in enumerate(loader):

    A = train(*batch)

    losses = np.append(losses,A.numpy())

    if idx % 500 == 0: print(idx)

    # Saving...
    if idx % save_idx == 0:

        print('Saving...')
        print(idx,'loss:',A.numpy())

        if idx == 0: continue

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

            predictions, predictions_origin, plotting = query(*plotBatch)

            predictions = predictions.numpy()[:,1] # Only need the 2nd column of values - these values represent the probablity of track being in class 1
            predictions_origin = predictions_origin.numpy()[0]

            # Post-processing step - convert x and y NN outputs back to physical values...
            predictions_origin[0] = predictions_origin[0]*6.
            predictions_origin[1] = (predictions_origin[1] - 0.5)*4.

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
                if predictions[kdx] < 0.5:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:blue',alpha=0.25)
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=0.25,color='k', marker='x')
                else:
                    plt.plot(points[kdx,:-1,0],points[kdx,:-1,1],color='tab:red')
                    plt.scatter(points_at_layers[kdx,:-1,0],points_at_layers[kdx,:-1,1],alpha=1.,color='k', marker='x')
                    nTracks[2] += 1
            
            plt.scatter(predictions_origin[0],predictions_origin[1],alpha=1.,color='tab:red', marker='x',s=100)

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
