//@author Sottithat Winyarat; winyarat@seas.upenn.edu
#include"util.h"



MatrixXd util::extractFeatures(string imagePath, vector<orb::FASTfeature>& ffeats, int& FASTradius, double& FASTthreshold, map<int, tuple<MatrixXi, MatrixXi>>& BRIEFtable, int scaleLevel ){
	//1. read in PNG image
	orb::rawRGB rawim;
	orb::readPNG(imagePath, rawim);

	//2. convert to gray scale image
	MatrixXd grayIm = orb::rbg2grayMat(rawim.image,rawim.width,rawim.height);

	//3. Smooth 
    MatrixXd kernel = orb::gaussianKernel2D(5,sqrt(2)*2); //kernel of width 3 and std of sqrt(2)
	MatrixXd smoothedIm = orb::conv2d(grayIm, kernel);
	smoothedIm = orb::subsample(smoothedIm);
	MatrixXd outIm(smoothedIm);

	//4. Extract FAST features
	double FASTscale = 1.0; //scale is with respect to the initial subsampled image since we discard the raw images
	orb::extractFASTfeats(ffeats, smoothedIm, FASTscale, FASTradius, FASTthreshold); 


	for(int i = 0; i < scaleLevel - 1; i++){
		FASTscale = FASTscale/2.0;
		smoothedIm = orb::conv2d(smoothedIm, kernel);
		smoothedIm = orb::subsample(smoothedIm);
		orb::extractFASTfeats(ffeats, smoothedIm, FASTscale, FASTradius, FASTthreshold); 
	}

	//5. Assign BRIEF descriptors to FAST features
	orb::BRIEFdescriptors(ffeats, outIm, BRIEFtable); 

	//6. return the smoothed gray image
	return outIm;
}

//build BRIEF test-pairs table: a map from feature orientation ID to test pairs 
map<int, tuple<MatrixXi, MatrixXi>> util::buildBRIEFtestLookUp(int& FASTradius){
	
	MatrixXd left(2, orb::BITNUMBER);
	MatrixXd right(2, orb::BITNUMBER);
	tie(left, right) = orb::generateTestPairs(FASTradius, orb::BITNUMBER); //Gaussian generated test pairs
	map<int, tuple<MatrixXi, MatrixXi>> BRIEFtable = orb::buildTestPairsLookup(left,right); //a lookup table of rotated test pairs

	return BRIEFtable;
}

//load rectified camera intrinsic parameters from a txt file 
//IMPORTANT REMARK: imscale indicates the proportion of stereo image size to the raw image size.
map<string, double> util::loadCamIntrinsics(string fname, double imscale){
	map<string, double> out;
	ifstream file(fname);
	string line;
	//find the line corresponding to a given camera ID
	if(file.is_open()){
		cout << "Loading camera intrinsics from file...\n";
		string key = "";
		double val = 0.0;
		while ( getline (file,line) ){
			istringstream iss(line);
			iss >> key;
			iss >> val;
			out[key] = val*imscale;
		}
		file.close();
	}
	else{
		cerr << "load camerca intrinsic failed.\n" ;
	}
	
	return out;		
}


//load IMU noise parameters from file
map<string, double> util::loadIMUcalibration(string fname){

	map<string, double> out;
	ifstream file(fname);
	string line;
	//find the line corresponding to a given camera ID
	if(file.is_open()){
		cout << "loading IMU calibration parameters...\n";
		string key = "";
		double val = 0.0;
		while ( getline (file,line) ){
			istringstream iss(line);
			iss >> key;
			iss >> val;
			out[key] = val;
		}
		file.close();
	}
	else{
		cerr << "load IMU calibrationnoise parameters failed.\n" ;
	}
	return out;		
}

//load IMU measurements from file into a vector
vector<util::IMU> util::readIMUfromFile(string fname){
	vector<util::IMU> out;
	ifstream file(fname);
	string line;
	string s;
	double temp;
	
	if(file.is_open()){
		cout << "Reading IMU measurements from file...\n";
		while ( getline (file,line) ){
			//skip csv file header
			if(std::isdigit(line[0])){ 
				istringstream iss(line);
				vector<double> tokens;
				//collect comma-separated token
				while(getline(iss,s, ','  )){
					temp = stod(s);
					tokens.push_back(temp);
				}

				//collect imu reading packet
				double timestamp = tokens[0];
				Eigen::Vector3d gyro{tokens[1],tokens[2],tokens[3]};
				Eigen::Vector3d accel{tokens[4],tokens[5],tokens[6]};
				out.push_back(util::IMU(timestamp, accel, gyro));
			}
			
		}
		file.close();
	}
	else{
		cerr << "load IMU measurements failed.\n" ;
	}
	return out;	
}

//load the 2 cameras' extrinsics(pose with respect to IMU)
map<string, Eigen::MatrixXd> util::loadCamExtrinsics(string fname){
	map<string, Eigen::MatrixXd> out; 
	
	Eigen::Matrix<double, 4, 4, RowMajor> R;
	
	ifstream file(fname);
	string line;
	//find the line corresponding to a given camera ID
	if(file.is_open()){
		cout << "Loading camera extrinsics from file...\n";
		string key = "";
		double val = 0.0;
		while ( getline (file,line) ){
			istringstream iss(line);
			iss >> key;
			double data[16];
			int i = 0;
			while(iss){
				double temp;
				iss >> data[i];
				i++;
			}
			Eigen::Matrix<double, 4, 4, RowMajor> R(data);
			out[key] = R;
		}
		file.close();
	}
	else{
		cerr << "load camerca extrinsics failed.\n" ;
	}
	return out;		
}

//hat operator mapping 3-vector to its associated skew-symmetric matrix
Eigen::Matrix<double,3,3> util::hat(Eigen::Vector3d x){
	Eigen::Matrix<double,3,3> out;
	out << 0,-x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
	return out;
}

//extract timestamp from image's name 
double util::extract_timestamp(string path){
	string output = std::regex_replace(
        path,
        std::regex("[^0-9]*([0-9]+).*"),
        std::string("$1")
        );

	return std::stod(output);
}



//load rectified camera intrinsic parameters from a txt file 
//IMPORTANT REMARK: imscale indicates the proportion of stereo image size to the raw image size.
vector<Eigen::Vector3d> util::loadUVD(string fname){
	vector<Eigen::Vector3d> out;
	ifstream file(fname);
	string line;
	//find the line corresponding to a given camera ID
	if(file.is_open()){
		cout << "Loading uvds from file...\n";
		double u = 0.0;
		double v = 0.0;
		double d = 0.0;
		while ( getline (file,line) ){
			Eigen::Vector3d uvd;
			istringstream iss(line);
			iss >> u;
			iss >> v;
			iss >> d;
			uvd << u,v,d;
			out.push_back(uvd);
		}
		file.close();
	}
	else{
		cerr << "load uvds failed.\n" ;
	}
	
	return out;		
}







