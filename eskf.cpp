#include"eskf.h"

using namespace eskf;
using namespace std;

//write State to std out 
std::ostream& eskf::operator<<(std::ostream& os, const State& x)
{
    os << "position: " << x.p.transpose() << '\n' <<
    	"velocity: " << x.v.transpose() << '\n' <<
    	"quaternion: " << x.q << '\n' <<
    	"accel bias: " << x.a_b.transpose() << '\n' <<
    	"gyro bias: " << x.w_b.transpose() << '\n' <<
    	"gravity: " << x.g.transpose() << '\n' ;

    return os;
}

//update error-state covariance
eskf::CovarMatrix eskf::update_error_state_covar(eskf::CovarMatrix& P, const eskf::State& x, const util::IMU& imu, map<string, double>& noise_params, double dt){
	
	//0. Compute matrix components
	Eigen::Vector3d rotvec = (imu.gyro - x.w_b)*dt;
	double ang = rotvec.norm();
	rotvec.normalize();
	Eigen::Quaterniond quat_temp(AngleAxisd(ang, rotvec));
	Eigen::MatrixXd R_trans = (quat_temp.toRotationMatrix()).transpose();
	Eigen::MatrixXd skewsym = util::hat(imu.accel - x.a_b);
	Eigen::MatrixXd R = x.q.toRotationMatrix();
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3,3);
	
	//1. Build error-state Jacobian
	Eigen::MatrixXd Fx = Eigen::MatrixXd::Zero(eskf::state_dim,eskf::state_dim);
	Fx.block(0,0,3,3) = I;
	Fx.block(0,3,3,3) = I*dt;
	Fx.block(3,3,3,3) = I;
	Fx.block(3,6,3,3) = -R*skewsym*dt;
	Fx.block(3,9,3,3) = -R*dt;
	Fx.block(3,15,3,3) = I*dt;
	Fx.block(6,6,3,3) = R_trans;
	Fx.block(6,12,3,3) = -I*dt;
	Fx.block(9,9,3,3) = I;
	Fx.block(12,12,3,3) = I;
	Fx.block(15,15,3,3) = I;

	//2. Build perturbation Jacobian
	Eigen::MatrixXd Fi = Eigen::MatrixXd::Zero(18,12);
	Fi.block(3,0,3,3) = I;
	Fi.block(6,3,3,3) = I;
	Fi.block(9,6,3,3) = I;
	Fi.block(12,9,3,3) = I;

	//3. Build noise Covariance
	double an = noise_params["accelerometer_noise_density"];
	double wn = noise_params["gyroscope_noise_density"];
	double aw = noise_params["accelerometer_random_walk"];
	double ww = noise_params["gyroscope_random_walk"];

	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(12,12);
	Q.block(0,0,3,3) = an*an*dt*dt*I;
	Q.block(3,3,3,3) = wn*wn*dt*dt*I;
	Q.block(6,6,3,3) = aw*aw*dt*I;
	Q.block(9,9,3,3) = ww*ww*dt*I;

	return Fx*P*(Fx.transpose()) + Fi*Q*(Fi.transpose());
} 

//correct error-state and its covariance by incorporting one image measurement(temporal correspondence)
tuple<eskf::State, eskf::CovarMatrix, Eigen::Vector2d> eskf::measurement_update(eskf::State& nominal_state, eskf::CovarMatrix& error_state_covar, Eigen::Vector3d uvd, Eigen::Vector3d Pw, double inlierEpsilon, MatrixXd imageCovar, bool covarianceReset ){
	//0. Compute the innovation term (aka measurement-model discrepency)
	MatrixXd R_trans = (nominal_state.q.toRotationMatrix()).transpose();
	Vector3d Pc = R_trans*(Pw- (nominal_state.p)); //map scene point in world frame to camera frame
	double u_pix = uvd(0); //normalized pixel coords
	double v_pix = uvd(1);
	Eigen::Vector2d innovation;
	innovation << u_pix - (Pc(0)/Pc(2)), v_pix - (Pc(1)/Pc(2));

	//1. Accept inliers
	if(innovation.norm() < inlierEpsilon){

		//2. Build partial dz/dP
		Eigen::MatrixXd partZpartP(2,3);
		partZpartP << 1.0/Pc(2), 0, -Pc(0)/(Pc(2)*Pc(2)),
					  0, 1.0/Pc(2), -Pc(1)/(Pc(2)*Pc(2)); 

		//3. Build partial dP/d(delta_Th)
		Eigen::Matrix3d  Pc_skewsym = util::hat(Pc);
		Eigen::MatrixXd partZpartdTh = partZpartP*Pc_skewsym;

		//4. Build partial dP/d(delta_p)
		Eigen::MatrixXd partZpartdp =  -partZpartP*R_trans;

		//5. Assemble error-state Jacobian H
		Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(2,3);
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2,18);
		H.block(0,0,2,3) = partZpartdp; //partial wrt to nominal position
		H.block(0,6,2,3) = partZpartdTh; //partial wrt to nominal orientation

		//6. Compute the Kalman Gain K
		MatrixXd S = H*error_state_covar*(H.transpose()) + imageCovar; //2x2
		MatrixXd K = error_state_covar*(H.transpose())*(S.inverse()); //18x2

		//7. Compute the corrected error-state dx, whose apriori mean is 0 
		VectorXd delta_x = K*innovation;
		eskf::State dx( delta_x);

		//8. Update covariance of the error-state(by reference)
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(18,18);
		Eigen::MatrixXd temp = I - (K*H);
		eskf::CovarMatrix next_error_state_covar = temp*error_state_covar*(temp.transpose()) + K*imageCovar*(K.transpose());

		//9. Recover the true state by nominal-error state composition(by reference)
		eskf::State next_nominal_state(nominal_state + dx);

		//10. Reset error-state covariance
		if(covarianceReset){
			eskf::resetCovariance(next_error_state_covar, delta_x.block(6,0,3,1));
		}
		
		return make_tuple(next_nominal_state, next_error_state_covar, innovation);
	}//end if

	return make_tuple(nominal_state, error_state_covar, innovation); //return the innovation vector for image noise analysis
}


//reset the error-state covariance 
void eskf::resetCovariance(eskf::CovarMatrix& error_state_covar, Eigen::Vector3d dtheta){
	//0. Compute partial-reset-partial-error
	Eigen::Matrix3d partRpartdx =  Eigen::Matrix3d::Identity(3,3) - 0.5*util::hat(dtheta);
	//1. Build the reset jacobian G
	eskf::CovarMatrix G = Eigen::MatrixXd::Identity(eskf::state_dim, eskf::state_dim);
	G.block(6,6,3,3) = partRpartdx;
	//2. reset
	error_state_covar  =  G*error_state_covar*(G.transpose());
}

