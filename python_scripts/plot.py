import numpy as np 
from numpy import loadtxt
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

def alignTime(a, b):
	if(a[0,0] > b[0,0] ):
		ref = a
		var = b

	else:
		ref = b
		var = a

	refLength = ref.shape[0]
	varLength = var.shape[0]
	i = 0
	j = 0 
	A = []
	B = []
	while(i < refLength and j+1 < varLength):
		while(ref[i,0] > var[j,0] and j < varLength-1):
			j += 1

		A.append(ref[i,:])
		B.append(var[j,:])
		i+= 1
	
	(A,B) = (np.array(A),np.array(B))

	#shift time to origin and scale to sec
	A[:, 0] = A[:, 0] - A[0, 0]
	B[:, 0] = B[:, 0] - B[0, 0]


	return (A,B)


######################################################################


#load machine hall  data
mission = "01"
groundtruth_mh = np.genfromtxt("../dataset/"+ "mh" + mission + "/state_groundtruth_estimate0/"+ 'data.csv', delimiter=',', dtype='float64', skip_header=1)
estimate_mh = loadtxt("../out_mh" + mission + ".txt" );


#trim and align timestamp 
(groundtruth_mh,estimate_mh) = alignTime(groundtruth_mh, estimate_mh)


#extract machine hall  groundtruths
groundtruth_mh_timestamp = groundtruth_mh[:, 0];
groundtruth_mh_position = groundtruth_mh[:, 1:4]
groundtruth_mh_quat = groundtruth_mh[:, 4:8]
groundtruth_mh_velocity= groundtruth_mh[:, 8:11]

#extract machine hall  estimates
estimate_mh_timestamp = estimate_mh[:, 0];
estimate_mh_position = estimate_mh[:, 1:4];
estimate_mh_velocity = estimate_mh[:, 4:7];
estimate_mh_quat = estimate_mh[:, 7:];

#convert quaternions to rotation vector
groundtruth_mh_angleAxis= np.empty((groundtruth_mh_quat.shape[0],3));
for i in range(0,groundtruth_mh_quat.shape[0]):
	rvec = R.from_quat(groundtruth_mh_quat[i,:]).as_rotvec();
	groundtruth_mh_angleAxis[i] = rvec;

estimate_mh_angleAxis= np.empty((estimate_mh_quat.shape[0],3));
for i in range(0,estimate_mh_quat.shape[0]):
	rvec = R.from_quat(estimate_mh_quat[i,:]).as_rotvec();
	estimate_mh_angleAxis[i] = rvec;




#plot positions
fig = plt.figure()
plt.plot(groundtruth_mh_timestamp, groundtruth_mh_position[:,0], 'royalblue');
plt.plot(groundtruth_mh_timestamp, groundtruth_mh_position[:,1], 'purple');
plt.plot(groundtruth_mh_timestamp, groundtruth_mh_position[:,2], 'slategrey');
plt.title("Machine Hall " + mission +  " Groundtruth: Position")
plt.xlabel("elasped time (nanosecond)")
plt.ylabel("position (m)")
plt.legend(["x", "y", "z"]);

fig = plt.figure()
plt.plot(estimate_mh_timestamp,estimate_mh_position[:,0], 'rosybrown');
plt.plot(estimate_mh_timestamp,estimate_mh_position[:,1], 'firebrick');
plt.plot(estimate_mh_timestamp,estimate_mh_position[:,2], 'coral');
plt.title("Machine Hall " + mission +  " Estimate: Position")
plt.xlabel("elasped time (nanosecond)")
plt.ylabel("position (m)")
plt.legend(["x", "y", "z"]);

#plot velocities
fig = plt.figure()

plt.plot(groundtruth_mh_timestamp,groundtruth_mh_velocity[:,0], 'royalblue');
plt.plot(groundtruth_mh_timestamp,groundtruth_mh_velocity[:,1], 'purple');
plt.plot(groundtruth_mh_timestamp,groundtruth_mh_velocity[:,2], 'slategrey');
plt.title("Machine Hall " + mission +  " Groundtruth: Velocity")
plt.xlabel("elasped time (nanosecond)")
plt.ylabel("velocity (m/s)")
plt.legend(["x", "y", "z"]);

fig = plt.figure()
plt.plot(estimate_mh_timestamp,estimate_mh_velocity[:,0], "rosybrown");
plt.plot(estimate_mh_timestamp,estimate_mh_velocity[:,1], 'firebrick');
plt.plot(estimate_mh_timestamp,estimate_mh_velocity[:,2], 'coral');
plt.title("Machine Hall " + mission +  " Estimate: Velocity")
plt.xlabel("elasped time (nanosecond)")
plt.ylabel("velocity (m/s)")
plt.legend(["x", "y", "z"]);

#plot rotation angle
angs_groundtruth = np.sum(np.abs(groundtruth_mh_angleAxis)**2,axis=-1)**(1./2)
fig = plt.figure()
plt.plot(groundtruth_mh_timestamp, angs_groundtruth, 'royalblue');
plt.ylabel("rotation angle (radian)")
plt.xlabel("elasped time (nanosecond)")
plt.title("Machine Hall " + mission +  " Groundtruth: Rotation Angle")

estimate_mh_angleAxis = -estimate_mh_angleAxis
angs_estimate = np.sum(np.abs(estimate_mh_angleAxis)**2,axis=-1)**(1./2)
fig = plt.figure()
plt.plot(estimate_mh_timestamp, angs_estimate, 'rosybrown');
plt.ylabel("rotation angle (radian)")
plt.xlabel("elasped time (nanosecond)")
plt.title("Machine Hall " + mission +  " Estimate: Rotation Angle")


plt.show();
