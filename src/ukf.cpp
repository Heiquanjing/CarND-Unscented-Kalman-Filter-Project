#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  //
  is_initialized_ = false;

  previous_timestamp_ = 0;

  //state dimensions
  n_x_ = 5;
  n_aug_ = 7;
  n_z_laser_ = 2;
  n_z_radar_ = 3;
  lambda_ = 3 - n_aug_;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  //initial predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_pred_.fill(0.0);

  //set weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for(int i=1; i<2*n_aug_+1; i++){
    weights_(i) = 0.5 / (lambda_ +n_aug_);
  }

  //set laser measurement matrix
  H_ = MatrixXd(n_z_laser_, n_x_);
  H_.fill(0.0);
  H_(0,0) = 1;
  H_(1,1) = 1;

  //set laser measurement noise matrix
  R_laser_ = MatrixXd(n_z_laser_, n_z_laser_);
  R_laser_ << std_laspx_*std_laspx_,  0,
              0,                      std_laspy_*std_laspy_;

  //set radar measurement noise matrix
  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_*std_radr_,  0,                        0,
              0,                    std_radphi_*std_radphi_,  0,
              0,                    0,                        std_radrd_*std_radrd_;


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if(!is_initialized_){
    // set state vector
    x_.fill(0.0);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho, phi, rho_dot;
      rho = meas_package.raw_measurements_[0];
      phi = meas_package.raw_measurements_[1];
      rho_dot = meas_package.raw_measurements_[2];

      x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      double px, py;
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];

      x_ << px, py, 0, 0, 0;
    }

    previous_timestamp_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  //compute the time elapsed between the current and previous measurements
  float delta_t = (meas_package.timestamp_-previous_timestamp_)/1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(delta_t);
  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  } else if(meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_,2*n_aug_+1);
  Xsig_aug_.fill(0.0);
  GenerateSigmaPoints(&Xsig_aug_);
  SigmaPointPrediction(&Xsig_aug_, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  VectorXd z_ = meas_package.raw_measurements_;
  VectorXd y_ = z_ - H_ * x_;
  MatrixXd S_ = H_ * P_ * H_.transpose() + R_laser_;
  MatrixXd K_ = P_ * H_.transpose() * S_.inverse();

  x_ = x_ + (K_ * y_);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K_ * H_) * P_;

  NIS_laser_ = y_.transpose() * S_.inverse() * y_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  MatrixXd Zsig_(n_z_radar_, 2*n_aug_+1);
  Zsig_.fill(0.0);
  VectorXd z_pred_(n_z_radar_);
  z_pred_.fill(0.0);
  MatrixXd S_pred_(n_z_radar_, n_z_radar_);
  S_pred_.fill(0.0);
  VectorXd z_ = meas_package.raw_measurements_;

  PredictRadarMeasurement(&Zsig_, &z_pred_, &S_pred_);
  UpdateRadarState(z_, Zsig_, z_pred_, S_pred_);

}


void UKF::GenerateSigmaPoints(MatrixXd *Xsig_aug_){
  VectorXd x_aug_ = VectorXd(n_aug_);

  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  // create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points
  (*Xsig_aug_).col(0)  = x_aug_;
  for (int i = 0; i < n_aug_; i++){
    (*Xsig_aug_).col(i+1) = x_aug_ + sqrt(lambda_+n_aug_)*L.col(i);
    (*Xsig_aug_).col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_)*L.col(i);
  }
}

void UKF::SigmaPointPrediction(MatrixXd *Xsig_aug_, double delta_t){
  for (int i = 0; i < 2*n_aug_+1; i++)
  {
    double p_x = (*Xsig_aug_)(0,i);
    double p_y = (*Xsig_aug_)(1,i);
    double v = (*Xsig_aug_)(2,i);
    double yaw = (*Xsig_aug_)(3,i);
    double yawd = (*Xsig_aug_)(4,i);
    double nu_a = (*Xsig_aug_)(5,i);
    double nu_yawdd = (*Xsig_aug_)(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance(){
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::PredictRadarMeasurement(MatrixXd *Zsig_, VectorXd *z_pred_, MatrixXd *S_pred_){
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    (*Zsig_)(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    (*Zsig_)(1,i) = atan2(p_y,p_x);                                 //phi
    (*Zsig_)(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  for (int i=0; i < 2*n_aug_+1; i++) {
      (*z_pred_) += weights_(i) * (*Zsig_).col(i);
  }

  //measurement covariance matrix S

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = (*Zsig_).col(i) - *z_pred_;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    *S_pred_ += weights_(i) * z_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  *S_pred_ += R_radar_;
}

void UKF::UpdateRadarState(VectorXd z_, MatrixXd Zsig_, VectorXd z_pred_, MatrixXd S_pred_){
  MatrixXd Tc(n_x_, n_z_radar_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  //Kalman gain K;
  MatrixXd K_ = Tc * S_pred_.inverse();

  //residual
  VectorXd z_diff = z_ - z_pred_;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K_ * z_diff;
  P_ = P_ - K_ * S_pred_ * K_.transpose();
}