#include"util.h"
#include"eskf.h"
#include<filesystem>
#include<set>
#include<boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;


int main(int argc, char **argv){

	///////////////////////////
	/////  program options  ///
	///////////////////////////
	string mission = "mh01";
	int num_images = 1000; // number of stereo image pairs to process
	double inlier_epsilon = 0.05; 
	double FASTthreshold = 60;
	int FASTradius = 11;
	int FASTscaleLevel = 2;
	bool saveim = false;
	bool covarianceReset = true;
	double measurement_covar_factor = 1.0e-10;
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("mission", po::value<std::string>(&mission), "enter path to mission number; ex. mh01")
	    ("inlier_epsilon", po::value<double>(&inlier_epsilon), "enter inlier threshold value")
	    ("num_images", po::value<int>(&num_images), "enter the number of images to process")
	    ("saveim", po::value<bool>(&saveim), "save images of correspondences to directory?")
	    ("covarianceReset", po::value<bool>(&covarianceReset), "reset error covariance in measurement update?")
	    ("FASTradius", po::value<int>(&FASTradius), "enter the FAST radius") 
	    ("FASTthreshold", po::value<double>(&FASTthreshold), "enter the FAST threshold")
	    ("FASTscaleLevel", po::value<int>(&FASTscaleLevel), "enter the FAST scale level")
	    ("measurement_covar_factor", po::value<double>(&measurement_covar_factor), "measurement covariance factor");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);  

  	///////////////////////////////////////
  	//// Body2World frame transformation //
  	///////////////////////////////////////
  	Eigen::Vector3d t_WB;
  	Eigen::Quaterniond q_WB = Eigen::Quaterniond::Identity();
  	if(mission == "mh01"){
  		//Taken from the initial groundtruth 
  		t_WB << 4.688319,	-1.786938, 0.783338;
	  	q_WB = Eigen::Quaterniond(0.534108,	-0.153029,	-0.827383,-0.082152);
  	}
  	else if(mission == "mh03"){
  		t_WB << 4.631117,	-1.786812,	0.577113;
	  	q_WB = Eigen::Quaterniond(0.606317,	-0.156367,	-0.776345,	-0.072229);
  	}
  	else if(mission == "mh04"){
		t_WB << 4.677066,	-1.749440,	0.568567;
	  	q_WB = Eigen::Quaterniond(0.240749,	-0.761130,	-0.355916,	-0.485843);
  	}
  	else{
  		std::cerr << "INVALID MISSION NUMBER: Please choose among missions mh01, mh03, and mh04 \n";
  		abort();
  	}
  	
  	Eigen::MatrixXd R_WB = q_WB.toRotationMatrix();
  	Eigen::MatrixXd T_WB = Eigen::MatrixXd::Identity(4,4);
  	T_WB.block(0,0,3,3) = R_WB;
  	T_WB.block(0,3,3,1) = t_WB;

	///////////////////////////////
	//// Stereo Directory Setup  //
	// ////////////////////////////
	string leftStereoPath = "./dataset/" + mission + "/cam0_rectified";
	string rightStereoPath = "./dataset/" + mission + "/cam1_rectified";
	string imuMeasurementPath = "./dataset/" + mission + "/imu0/data.csv";
	string imuNoisePath = "./dataset/" + mission + "/imu_params.txt";
	string cameraIntrinsicsPath = "./dataset/" + mission + "/cam_intrinsics_rectified.txt";
	string cameraExtrinsicsPath = "./dataset/" + mission + "/cam_extrinsics.txt";  
	//sort paths to stereo images by timestamp 
   	set<fs::path> sorted_left_stereo;
   	set<fs::path> sorted_right_stereo;
  	for (auto &entry : fs::directory_iterator(leftStereoPath)){
  		sorted_left_stereo.insert(entry.path());
  	}
  	for (auto &entry : fs::directory_iterator(rightStereoPath)){
  		sorted_right_stereo.insert(entry.path());
  	}
  	auto left_stereo_iter = sorted_left_stereo.begin();
  	auto right_stereo_iter = sorted_right_stereo.begin();

  	// std::advance(left_stereo_iter,500);
  	// std::advance(right_stereo_iter,500);



	
	//////////////////////////////
	//Camera and Stereo Setup////
	//////////////////////////////
	MatrixXd leftIm;
	MatrixXd rightIm;
	MatrixXd Im_00; //image at time t
	MatrixXd Im_01; //image at time t+1
	const double imageScale = 0.5; //images are all subsampled once 
	const int stereoHalfWindow = 3; //number of verticle pixels away from epipolar line
	int temporalWindow = 20; //verticle pixel window centered at current(t) feature's pixel coordinate. 
	//We search a temporal correspondence within this band in the next(t+1) image.
	auto camera_intrinsics = util::loadCamIntrinsics(cameraIntrinsicsPath, imageScale);
	auto camera_extrinsics = util::loadCamExtrinsics(cameraExtrinsicsPath);
	Eigen::Matrix3d R_LB = ((camera_extrinsics["cam0"]).block(0,0,3,3)).transpose(); // extract rotation mapping from IMU to left cam
	Eigen::Quaterniond q_LB(R_LB);
	Eigen::Vector3d t_BL = (camera_extrinsics["cam0"]).block(0,3,3,1); //translation vector from body to left cam
	double first_stereo_time = util::extract_timestamp((*left_stereo_iter).filename());
	double next_stereo_time = util::extract_timestamp((*next(left_stereo_iter)).filename());
    std::vector<orb::StereoKeypt> stereo_00; //stereo keypoints at time t
    std::vector<orb::StereoKeypt> stereo_01;//stereo keypoints at time t+1
    std::vector<orb::FASTfeature> feats_left_01;//Features found in the left image at time t+1
    std::vector<orb::FASTfeature> feats_right_01;//Features found in the right image at time t+1
    std::vector<orb::TemporalCorrespondence> corrs; //temporal correspondences
    map<int, tuple<MatrixXi, MatrixXi>> BRIEFtable =  util::buildBRIEFtestLookUp(FASTradius);
   	leftIm =  util::extractFeatures(*left_stereo_iter, feats_left_01, FASTradius, FASTthreshold, BRIEFtable, FASTscaleLevel);
	rightIm = util::extractFeatures(*right_stereo_iter, feats_right_01, FASTradius, FASTthreshold, BRIEFtable, FASTscaleLevel);
	Im_00 = leftIm;	
	stereo_01 = orb::stereoMatch(feats_left_01,feats_right_01,stereoHalfWindow ,camera_intrinsics);
	left_stereo_iter++;
	right_stereo_iter++;

	
	///////////////
	//IMU Setup////
	///////////////
	auto imus = util::readIMUfromFile(imuMeasurementPath);
	auto imu_noise_params = util::loadIMUcalibration(imuNoisePath);
	auto imu_iter = imus.begin();
	for(auto&x: imus){
		//Map all imu measurements into left cam frame 
		x.accel = R_LB*x.accel;
		x.gyro = R_LB*x.gyro;
	}
	
	///////////////
	//output file//
	///////////////
	ofstream outfile("log/out_"+ mission + ".txt");

	//1. Set up counters and data containers for pose collection 
	int imageCount = 0;
	int imuCount = 0;
	double t = 0;//current time 
	double dt = 0;//time differential
	double last_timestamp = first_stereo_time;
	
	//2. Push imu past first stereo images and pretend to have processed the first stereo image
	while(imu_iter->timestamp < first_stereo_time){
		imu_iter++;
	}
	

	//3. Initialize error-state covariance, its covariance, and image(measurement) covariances
	Eigen::Vector3d zero{0,0,0};
	Eigen::Vector3d gravity = R_LB*(T_WB.transpose().block(0,2,3,1)); //map negative z-axis in world to left camera frame
	gravity.normalize();
	gravity = -9.8*gravity;//normalized and reweight 
	eskf::State nominal_state(zero,zero, Quaterniond::Identity() , zero, zero, gravity);
	Eigen::VectorXd covarVec(18,1);
	covarVec << 0, 0, 0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0, 0, 0;
	eskf::CovarMatrix error_state_covar = covarVec.asDiagonal() ; 
	Eigen::MatrixXd image_measurement_covar = measurement_covar_factor*Eigen::MatrixXd::Identity(2,2);	
	Eigen::MatrixXd image_measurement_covar_2 = image_measurement_covar*10;
	
	Eigen::MatrixXd last_R = nominal_state.q.toRotationMatrix(); //track previous pose
	Eigen::Vector3d last_t = nominal_state.p;
	bool stereo_processed = false;




	// 4. Loop through data
	while(true){

		if(imu_iter >= imus.end() || imageCount > num_images){
			break;
		}

		cout << "	Processing imu# " << imuCount << '\n';
		cout << "		Number of Images processed: " << imageCount << '\n';
		imuCount++;

		//a. Compute time differential dt
		t = std::min(imu_iter->timestamp, next_stereo_time);
		dt = (t-last_timestamp)*1e-9; //nanosec
		last_timestamp = t;

		//b. Process IMU measurements
		error_state_covar = eskf::update_error_state_covar(error_state_covar, nominal_state, *imu_iter,  imu_noise_params, dt);
		nominal_state.integrate_imu(*imu_iter,dt);

		// c. Process stereo image if it has arrived
		if(imu_iter->timestamp <= next_stereo_time){
			imu_iter++;
		}
		else{
			//d. match left and right stereo 
			stereo_processed = true;
			stereo_00 = stereo_01;
			feats_left_01.clear();
			feats_right_01.clear();
			leftIm = util::extractFeatures(*left_stereo_iter, feats_left_01, FASTradius, FASTthreshold, BRIEFtable, FASTscaleLevel);
			rightIm = util::extractFeatures(*right_stereo_iter, feats_right_01, FASTradius, FASTthreshold, BRIEFtable, FASTscaleLevel);
			cout << "	# of features extracted: " << feats_right_01.size() << '\n';
			stereo_01 = orb::stereoMatch(feats_left_01,feats_right_01,stereoHalfWindow ,camera_intrinsics);
			left_stereo_iter++;
			right_stereo_iter++;
			imageCount++;
			Im_00 = Im_01;
			next_stereo_time = util::extract_timestamp((*left_stereo_iter).filename());


			cout << "	# of stereo correspondences: " << stereo_01.size() << '\n';
			// to draw and save temporal correspondences
			if(saveim){
				Im_01 = leftIm;
				MatrixXd temp_Im_00(Im_00);
				MatrixXd temp_Im_01(Im_01);
				orb::drawTemporalCorr(temp_Im_00, temp_Im_01, corrs);
				orb::writePNG("images/"+to_string(imageCount)+ "_00.PNG", temp_Im_00);
				orb::writePNG("images/"+to_string(imageCount)+ "_01.PNG", temp_Im_01);
			}
			//e. match temporal
			corrs = orb::temporalMatch(stereo_00, stereo_01, (1.0+ nominal_state.v.norm())* temporalWindow);
			cout << "	# of temporal correspondences: " << corrs.size() << '\n';
			

			//f. perform the correction step by incorporating all temporal correspondences, rejecting outliers
			Vector2d inno;
			int inlier_count = 0;

			if(imageCount > 400){
				inlier_epsilon = 0.08;
				image_measurement_covar = image_measurement_covar_2;
			}


			for(auto& corr: corrs){
					Vector3d Pw = last_R* corr.current.getSceneCoords() + last_t; //map scenepoint to world frame
					std::tie(nominal_state, error_state_covar, inno) = eskf::measurement_update(nominal_state,error_state_covar, 
									   corr.next.getNormalizedCoords(), Pw, inlier_epsilon, image_measurement_covar , covarianceReset);
					if(inno.norm() < inlier_epsilon){
						inlier_count++;
					}
			}
			cout << "		Number of inliers: " << inlier_count << '\n';
			last_R = nominal_state.q.toRotationMatrix(); //corrected pose
			last_t = nominal_state.p;

		}//end stereo condition
		
		// write state to file
		if(outfile.is_open() && stereo_processed){
			outfile << std::setprecision(20) << last_timestamp + dt*1e+9 << ' ';
			Eigen::Vector3d pp = R_WB* (R_LB.transpose()* nominal_state.p + t_BL) + t_WB;
			outfile << pp(0) << ' ' <<  pp(1) << ' ' << pp(2) << ' '; //write global position
			Eigen::Vector3d vv = R_WB*(R_LB.transpose()* nominal_state.v);
			outfile << vv(0) << ' ' <<  vv(1) << ' ' << vv(2) << ' '; //write global velocity
			Eigen::Quaterniond qq =  q_WB*(nominal_state.q);
			outfile << qq.w() << ' ' <<  qq.x() << ' ' << qq.y() << ' ' << qq.z() <<  '\n'; //write global quaternion

		}
		
	}//end while

	outfile.close();

	return 0;
}
