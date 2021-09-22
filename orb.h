#ifndef ORB_H
#define ORB_H

#include"lodepng/lodepng.h"
#include<iostream>
#include<fstream>
#include<numeric>
#include<Eigen/Dense>
#include<algorithm>
#include<math.h>
#include<random>
#include<bitset>
#include<map>
#include<set>

using namespace Eigen;
using namespace std;
//define a ROWMAJOR matrix of image pixels
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixIM;
namespace orb{

	const int BITNUMBER = 256;// The number of bits to encode BRIEF descriptor with. 

	struct pixelCoord{
		int u ;
		int v;
		pixelCoord(int x, int y): u(x), v(y){};
		pixelCoord(): u(0), v(0){};
		Eigen::Vector2d toVector(){
			Eigen::Vector2d out;
			out << u,v;
			return out;
		}
	};

	struct rawRGB{
		vector<unsigned char> image;
		unsigned int width;
		unsigned int height;
		rawRGB(vector<unsigned char> im, unsigned int w, unsigned int h): image(im), width(w), height(h){};
		rawRGB(): image(), width(0), height(0) {};
	};

	struct ScaleSpaceImage{
		MatrixXd spatialImage; //2D matrix of smoothed pixel values;
		double sigma; //the sigma of Gaussian used to smooth this image
		double scale; //image resolution(aka. subsampling rate) relative to the base image
		ScaleSpaceImage(MatrixXd im, double s, double sc): spatialImage(im), sigma(s), scale(sc){};
		ScaleSpaceImage(): spatialImage(), sigma(1.0), scale(1.0){};
	};

	struct FASTfeature{
		int radius; //radius of FAST ring
		int u; //x-pixel
		int v; //y-pixel
		double harris; //harris corner measure value
		double scale; //resolution relative to base image
		int orientationID; //ID associated with a 30 degree increment bin. eg. ID == 0 means 0 < feature orientation  <= 30 degree 
		bitset<BITNUMBER> BRIEFdescriptor;
		int hdistance; //hamming distance between other feature descriptor and itself
		FASTfeature(int u, int v, int r, double s, double h, int o): u(u), v(v), radius(r), scale(s), harris(h),orientationID(o){};
		FASTfeature(): radius(0),u(0), v(0), harris(0), scale(0.0), orientationID(-1){}; 
	};

	struct StereoKeypt{
		pixelCoord lpix; //left image pixel
		pixelCoord rpix; //right image pixel
		double scale = 1.0; //proportion to the original raw image
		bitset<BITNUMBER> BRIEFdescriptor; // binary descriptor of left image feature
		double disparity; //ul - ur
		double X; //x coordinate in camera's frame
		double Y; // y coordinate in camera's frame
		double Z; //depth
		bool isClose; //whether a stereo keypoint is close or far. A close keypoint has depth that is less than 40xbaseline of stereo rig.
		double hdistance; //hamming distance between some other StereoKeypt's descriptor and itself. 
		int FASTradius;
		
		StereoKeypt(pixelCoord l, pixelCoord r, map<string, double> camIntrinsics, bitset<BITNUMBER> descrp, int FASTradius, double imscale): lpix(l), rpix(r), BRIEFdescriptor(descrp), FASTradius(FASTradius), scale(imscale){
			disparity = l.u-r.u;
			Z = (camIntrinsics["fx"]*camIntrinsics["bline"])/disparity;
			X = (l.u - camIntrinsics["cx"])*(Z/camIntrinsics["fx"]);
			Y = (l.v- camIntrinsics["cy"])*(Z/camIntrinsics["fy"]);
			isClose = Z < 40* camIntrinsics["bline"]; //As suggested in L.M Paz, "Large-scale 6-DOF SLAM with stereo in hand".
		
		};

		Vector3d getNormalizedCoords(){
			Vector3d out;
			out << X/Z, Y/Z, 1/(Z);
			return out;
		}

		Vector3d getSceneCoords(){
			Vector3d out;
			out << X, Y, Z;
			out = out/scale;
			return out;
		}
	};

	struct TemporalCorrespondence{
		StereoKeypt current; //keypoint at time t
		StereoKeypt next; //keypoint at time t+1
		double weight; // 1 minus the ratio of the hamming distance between the 2 Stereo keypts over total bit number
		TemporalCorrespondence(StereoKeypt c, StereoKeypt n, double w):current(c), next(n), weight(w){}; 
	};

	//image preprocessing methods
	vector<Matrix<double, 3, 4, RowMajor>> loadGroundTruths(const string& fname);
	void readPNG(const string& filename, rawRGB& raw);
	void writePNG(string filename, const MatrixXd& grayMat);
	MatrixIM rbg2grayMat(const vector<unsigned char>& image, int width, int height);
	Eigen::MatrixXd conv2d_separable(const Eigen::MatrixXd& baseIm, const Eigen::MatrixXd& oneDKernel);
	Eigen::MatrixXd conv1d(const Eigen::MatrixXd& baseIm, const Eigen::MatrixXd& kernel);
	MatrixXd conv2d(const MatrixIM& baseIm, const MatrixXd& kernel);
	MatrixXd gaussianKernel2D(int width, double sigma);
	MatrixXd subsample(MatrixXd& source);
	map<string, double> loadCamIntrinsics(string fname, int camID, double imscale);
	
	//FAST detector methods
	vector<pixelCoord> getNeighborOffsets(const int& radius);
	bool testAnchors(MatrixXd& image, const int& u, const int& v, vector<int>& anchors, const double& FASTthreshold, vector<orb::pixelCoord>& offsets, bool isLess, int quater);
	bool FASTtest(MatrixXd& image, const int& u, const int& v, const double& FASTthreshold, vector<orb::pixelCoord>& offsets, bool isLess);
	void extractFASTfeats(vector<orb::FASTfeature>& feats, MatrixXd& image, const double imscale, const int FASTradius, const double FASTthreshold);
	double SAD(MatrixXd& image, int u, int v, int patchWidth);
	double harrisMeasure(MatrixXd& image, int u, int v, int patchWidth, const MatrixXd& weight, double alpha);
	orb::FASTfeature suppressNonMax(MatrixXd& image, int u, int v, int FASTradius, double FASTthreshold, double imscale, vector<orb::pixelCoord>& offsets);
	void featureOrientationID(MatrixXd& image, orb::FASTfeature& feat);
	void drawFeatures(MatrixXd& image, vector<orb::FASTfeature> ffeats);
	void drawStereoCorr(MatrixXd& leftimage, MatrixXd& rightimage, orb::StereoKeypt corr);
	void drawTemporalCorr(MatrixXd& leftimage, MatrixXd& rightimage, vector<orb::TemporalCorrespondence> corrs);
	
	//BRIEF descriptors Methods
	void featurePatch2PNG(MatrixXd& image, orb::FASTfeature feat);
	tuple<MatrixXd, MatrixXd> generateTestPairs(int patchWidth, int N);
	map<int, tuple<MatrixXi, MatrixXi>> buildTestPairsLookup(MatrixXd& leftTests, MatrixXd& rightTests);
	void BRIEFdescriptors( vector<orb::FASTfeature>& feats, MatrixXd& image, map<int, tuple<MatrixXi, MatrixXi>>& lookupTable);
	int hammingDistance(bitset<BITNUMBER> a, bitset<BITNUMBER> b);
	
	//Feature matching methods
	vector<orb::StereoKeypt> stereoMatch(vector<orb::FASTfeature>& leftFeatures, vector<orb::FASTfeature>& rightFeatures, int window, map<string, double>& CamIntrinsics);
	vector<orb::TemporalCorrespondence> temporalMatch(vector<orb::StereoKeypt>& currKeypts, vector<orb::StereoKeypt>& nextKeypts, int window);
	vector<vector<double>> correspondences2scenePoints(vector<orb::TemporalCorrespondence>& corrs);
	void normalizeCorrespondenceWeight(vector<orb::TemporalCorrespondence>& corrs);
}
#endif
