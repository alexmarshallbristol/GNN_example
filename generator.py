'''
    
    Toy dataset generator.

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def generate_dataset(nTracks=20, plot=False):

	detector_planes = [7., 9.]

	if nTracks < 10:
		print("Minimum nTracks is",10)

	nTracks_B = np.random.randint(3, 6) # Between 3 and 6 tracks in the event should be from the B (1 B per event).
	# nTracks_B = 5

	nTracks_PV = nTracks - nTracks_B

	if plot:
		print(nTracks_B, nTracks_PV)

	# eta = -np.log(np.tan(theta/2.)) # Pseudorapidity

	eta =  np.random.uniform(low=2., high=5.0, size=1) # LHCb rough acceptance (https://lhcb.web.cern.ch/speakersbureau/html/PerformanceNumbers.html)
	eta = eta*np.sign(np.random.uniform(-1,1)) # Populate both sides of the beamline
	theta_dis = 2.*np.arctan(np.exp(-eta)) # rads
	x_dis = truncnorm.rvs(-1, 3, size=1)[0]*1.5+2. # Max displacement is 6.5
	# x_dis = truncnorm.rvs(-1, 1, size=1)[0]*0.5+4.5

	y_dis = x_dis*np.tan(theta_dis)[0]
	displaced_vertex = [x_dis,y_dis]

	points = np.empty((0,3,2))
	plot_out_to_x = 13.

	points_at_layers = np.empty((0,3,2))

	origin = np.empty((0,2))

	for idx in range(nTracks):


		eta =  np.random.uniform(low=2., high=5.0, size=1) # LHCb rough acceptance (https://lhcb.web.cern.ch/speakersbureau/html/PerformanceNumbers.html)
		eta = eta*np.sign(np.random.uniform(-1,1)) # Populate both sides of the beamline
		theta = 2.*np.arctan(np.exp(-eta)) # rads

		if idx < nTracks_PV:

			# Generate tracks from the PV - primary vertex (0,0)

			x = plot_out_to_x
			y = x*np.tan(theta)[0]
			points = np.append(points, [[[0.,0.],[x,y],[0.,0.]]], axis=0)
			values = [[[detector_planes[0],detector_planes[0]*np.tan(theta)[0]],[detector_planes[1],detector_planes[1]*np.tan(theta)[0]],[0.,0.]]]
			points_at_layers = np.append(points_at_layers, values, axis=0)
			origin = np.append(origin, [[0.,0.]], axis=0)
		else:

			# Generate tracks from the displaced vertex

			x = plot_out_to_x - displaced_vertex[0]
			y = x*np.tan(theta+theta_dis)[0]
			points = np.append(points, [[[displaced_vertex[0],displaced_vertex[1]],[x+displaced_vertex[0],y+displaced_vertex[1]],[1.,1.]]], axis=0)

			values_A = (detector_planes[0] - displaced_vertex[0])*np.tan(theta+theta_dis)[0]+displaced_vertex[1]
			values_B = (detector_planes[1] - displaced_vertex[0])*np.tan(theta+theta_dis)[0]+displaced_vertex[1]
			values = [[[detector_planes[0], values_A],[detector_planes[1], values_B],[1.,1.]]]
			points_at_layers = np.append(points_at_layers, values, axis=0)
			origin = np.append(origin, [displaced_vertex], axis=0)

	diff_x = detector_planes[1]-detector_planes[0]
	diff_y = points_at_layers[:,0,1] - points_at_layers[:,1,1] 
	theta = np.arctan(diff_y/(np.ones(np.shape(diff_y))*diff_x))
	inputs = np.empty((nTracks, 4))
	inputs[:,0] = points_at_layers[:,0,1]
	inputs[:,1] = theta

	origin[:,0] = origin[:,0]/6.
	origin[:,1] = origin[:,1]/4.+0.5

	inputs[:,2] = origin[:,0]
	inputs[:,3] = origin[:,1]

	# inputs[:,2] = np.concatenate((np.zeros(nTracks_PV), np.ones(nTracks_B)))
	# np.random.shuffle(inputs) # If you want to shuffle, need to shuffle everything - including plotting

	data = inputs[:,:-2]
	labels = inputs[:,-2:]

	if plot:
		for idx in range(nTracks):		
			if idx < nTracks_PV:
				plt.plot(points[idx,:,0],points[idx,:,1],color='tab:blue',alpha=0.25)
				plt.scatter(points_at_layers[idx,:,0],points_at_layers[idx,:,1],alpha=0.25,color='k', marker='x')
			else:
				plt.plot(points[idx,:,0],points[idx,:,1],color='tab:red')
				plt.scatter(points_at_layers[idx,:,0],points_at_layers[idx,:,1],alpha=1.,color='k', marker='x')

		plt.xlim(-1.5,10)
		plt.ylim(-3,3)
		plt.axhline(y=0,c='k',alpha=0.1)
		plt.axvline(x=0, c='k',alpha=0.1)
		plt.axvline(x=7, c='k',alpha=1)
		plt.axvline(x=9, c='k',alpha=1)
		plt.show()


	plotting = [points, points_at_layers]

	return data, labels, plotting


# generate_dataset()







