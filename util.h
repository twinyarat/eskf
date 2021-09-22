#ifndef UTIL_H
#define UTIL_H
#include"orb.h"
#include <regex>

namespace util{

	struct IMU{
		double timestamp;
		Eigen::Vector3d accel;
		Eigen::Vector3d gyro;

		IMU(double ts, Eigen::Vector3d acc, Eigen::Vector3d gyr):timestamp(ts),  accel(acc), gyro(gyr){};
	};


	MatrixXd extractFeatures(string imagePath, vector<orb::FASTfeature>& ffeats, int& FASTradius, double& FASTthreshold, map<int, tuple<MatrixXi, MatrixXi>>& BRIEFtable, int scaleLevel );
	map<int, tuple<MatrixXi, MatrixXi>> buildBRIEFtestLookUp(int& FASTradius);
	map<string, double> loadCamIntrinsics(string fname, double imscale);
	map<string, Eigen::MatrixXd> loadCamExtrinsics(string fname);
	map<string, double> loadIMUcalibration(string fname);
	vector<util::IMU> readIMUfromFile(string fname);
	vector<Eigen::Vector3d> loadUVD(string fname);

	Eigen::Matrix<double,3,3> hat(Eigen::Vector3d x);
	double extract_timestamp(string path);


}

#endif