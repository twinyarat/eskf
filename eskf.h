#ifndef ESKF_H
#define ESKF_H

#include"stdio.h"
#include"util.h"
#include<Eigen/Dense>
#include<iostream>

namespace eskf{
	const int state_dim = 18;
	typedef Matrix<double, state_dim, state_dim> CovarMatrix;


	struct State{
		friend std::ostream& operator<<(std::ostream &os, const State& n);
		Eigen::Vector3d p; //position
		Eigen::Vector3d v; //velocity
		Eigen::Quaterniond q; //orientation
		Eigen::Vector3d a_b; //accel bias
		Eigen::Vector3d w_b; //gyro bias
		Eigen::Vector3d g; //gravity

		//explicit constructor
		State(Eigen::Vector3d p, Eigen::Vector3d v, 
				Eigen::Quaterniond q, Eigen::Vector3d a_b,
				Eigen::Vector3d w_b, Eigen::Vector3d g): p(p), v(v), q(q), a_b(a_b), w_b(w_b), g(g){};
		
		//3-vector constructor(q is represented as a rotation vector r)
		//x := [p,v,r,a_b, w_b, g]'
		State(Eigen::Matrix<double, 18, 1> x):
				p(x.block(0,0,3,1)),
				v(x.block(3,0,3,1)),
				a_b(x.block(9,0,3,1)),
				w_b(x.block(12,0,3,1)),
				g(x.block(15,0,3,1))
				{	
					Eigen::Vector3d w = x.block(6,0,3,1);
					double ang = w.norm();
					w.normalize();
					Eigen::Quaterniond  qq(Eigen::AngleAxisd(ang, w));
					this->q = qq;
				};

		//composition of States
		State operator+(State& b){
			State out( p + b.p, v+b.v, q*b.q, 
					a_b + b.a_b, w_b + b.w_b, g+b.g);

			return out;
		};

		//nominal state update
		void integrate_imu(util::IMU& imu, const double& dt){
			Eigen::Vector3d temp = (this->q.toRotationMatrix()* (imu.accel - this->a_b)) + this->g;
			
			this->p = this->p + this->v*dt + 0.5*dt*dt*(temp) ;
			this->v = this->v + temp*dt;
			Eigen::Vector3d w = (imu.gyro -this->w_b)*dt;
			double ang = w.norm();
			w.normalize();
			Eigen::Quaterniond qtemp(Eigen::AngleAxisd(ang, w));
			this->q = this->q*qtemp;
			//accel bias, gyro bias, and gravity remain intact
		}

	};//end of State struct

	////////////////////////////////////////////
	//State and covariance propagation methods//
	////////////////////////////////////////////
	eskf::CovarMatrix update_error_state_covar(eskf::CovarMatrix& P, const eskf::State& x, 
													const util::IMU& imu, map<string, double>& noise_params,
													double dt);

	tuple<eskf::State, eskf::CovarMatrix, Eigen::Vector2d> measurement_update(eskf::State& nominal_state, eskf::CovarMatrix& error_state_covar, Eigen::Vector3d uvd, Eigen::Vector3d Pw, double InlierEpsilon, MatrixXd imageCovar, bool covarianceReset);

	void resetCovariance(eskf::CovarMatrix& error_state_covar, Eigen::Vector3d dtheta);
}
#endif