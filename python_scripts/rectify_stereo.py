#rectify and save raw EuroC images
import stereo_rectification

data = stereo_rectification.StereoDataSet("../dataset/mh04/");
data.stereo_intrinsics_to_file()
data.stereo_extrinsics_to_file();

for i in range(data.number_of_frames):
	print("rectifying image {}".format(i));
	stereoPair = data.rectify_stereo_pair(i);
