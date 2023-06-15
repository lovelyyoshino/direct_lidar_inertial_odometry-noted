/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/odom.h"

dlio::OdomNode::OdomNode(ros::NodeHandle node_handle) : nh(node_handle) {

  this->getParams(); //获得参数，主要是在cfg中读取的

  this->num_threads_ = omp_get_max_threads(); //获取最大的thread线程

  this->dlio_initialized = false;   // dlio标定是否初始化
  this->first_valid_scan = false;   //第一帧有效观测
  this->first_imu_received = false; //第一帧IMU数据
  if (this->imu_calibrate_) { // IMU是否已经标定，设置为true时候直接从cfg中读取
    this->imu_calibrated = false;
  } else {
    this->imu_calibrated = true;
  }
  this->deskew_status = false; //是否进行去畸变
  this->deskew_size = 0;       //去畸变的点云数量

  this->lidar_sub =
      this->nh.subscribe("pointcloud", 1, &dlio::OdomNode::callbackPointCloud,
                         this, ros::TransportHints().tcpNoDelay());
  this->imu_sub = this->nh.subscribe("imu", 1000, &dlio::OdomNode::callbackImu,
                                     this, ros::TransportHints().tcpNoDelay());

  this->odom_pub =
      this->nh.advertise<nav_msgs::Odometry>("odom", 1, true); //发布odom
  this->pose_pub = this->nh.advertise<geometry_msgs::PoseStamped>(
      "pose", 1, true); //发布pose
  this->path_pub =
      this->nh.advertise<nav_msgs::Path>("path", 1, true); //发布path
  this->kf_pose_pub = this->nh.advertise<geometry_msgs::PoseArray>(
      "kf_pose", 1, true); //发布关键帧的位姿
  this->kf_cloud_pub = this->nh.advertise<sensor_msgs::PointCloud2>(
      "kf_cloud", 1, true); //发布关键帧的点云
  this->deskewed_pub = this->nh.advertise<sensor_msgs::PointCloud2>(
      "deskewed", 1, true); //发布去畸变的点云

  this->publish_timer =
      this->nh.createTimer(ros::Duration(0.01), &dlio::OdomNode::publishPose,
                           this); //根据timer，发布pose

  this->T = Eigen::Matrix4f::Identity();       //初始化T
  this->T_prior = Eigen::Matrix4f::Identity(); //初始化T_prior，上一帧位姿
  this->T_corr = Eigen::Matrix4f::Identity(); //初始化T_corr,当前位姿

  this->origin = Eigen::Vector3f(0., 0., 0.);         //初始化原点
  this->state.p = Eigen::Vector3f(0., 0., 0.);        //初始化位置
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.); //初始化四元数
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.); //初始化线速度,机体坐标系下
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.); //初始化线速度,世界坐标系下
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.); //初始化角速度,机体坐标系下
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.); //初始化角速度,世界坐标系下

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);        //初始化lidar位置
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.); //初始化lidar四元数

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.; //初始化IMU的角速度
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.; //初始化IMU的线加速度
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_); //设置IMU的buffer
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = pcl::PointCloud<PointType>::ConstPtr(
      boost::make_shared<const pcl::PointCloud<PointType>>()); //初始化原始点云
  this->deskewed_scan = pcl::PointCloud<PointType>::ConstPtr(
      boost::make_shared<
          const pcl::PointCloud<PointType>>()); //初始化去畸变点云
  this->current_scan = pcl::PointCloud<PointType>::ConstPtr(
      boost::make_shared<const pcl::PointCloud<PointType>>()); //初始化当前点云
  this->submap_cloud = pcl::PointCloud<PointType>::ConstPtr(
      boost::make_shared<const pcl::PointCloud<PointType>>()); //初始化子图点云

  this->num_processed_keyframes = 0; //初始化处理的关键帧数量

  this->submap_hasChanged = true; //初始化子图是否改变,第一帧肯定改变
  this->submap_kf_idx_prev.clear(); //初始化上一帧的关键帧索引

  this->first_scan_stamp = 0.; //初始化第一帧点云的时间戳
  this->elapsed_time = 0.;     //初始化时间
  this->length_traversed;      //初始化长度

  this->convex_hull.setDimension(3);  //设置凸包的维度
  this->concave_hull.setDimension(3); //设置凹包的维度
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_); //设置凹包的阈值
  this->concave_hull.setKeepInformation(true); //设置凹包保留信息

  this->gicp.setCorrespondenceRandomness(
      this->gicp_k_correspondences_); //设置gicp的参数,这个值代表每次迭代时，随机选择的点对的数量
  this->gicp.setMaxCorrespondenceDistance(
      this->gicp_max_corr_dist_); //设置gicp的参数,这个值代表两个点云中对应点之间的最大距离
  this->gicp.setMaximumIterations(
      this->gicp_max_iter_); //设置gicp的参数,这个值代表最大迭代次数
  this->gicp.setTransformationEpsilon(
      this->gicp_transformation_ep_); //设置gicp的参数,这个值代表两次迭代之间的最小差异
  this->gicp.setRotationEpsilon(
      this->gicp_rotation_ep_); //设置gicp的参数,这个值代表两次迭代之间的最小旋转差异
  this->gicp.setInitialLambdaFactor(
      this->gicp_init_lambda_factor_); //设置gicp的参数,这个值代表初始lambda因子

  this->gicp_temp.setCorrespondenceRandomness(
      this->gicp_k_correspondences_); //设置gicp的参数,这个值代表每次迭代时，随机选择的点对的数量
  this->gicp_temp.setMaxCorrespondenceDistance(
      this->gicp_max_corr_dist_); //设置gicp的参数,这个值代表两个点云中对应点之间的最大距离
  this->gicp_temp.setMaximumIterations(
      this->gicp_max_iter_); //设置gicp的参数,这个值代表最大迭代次数
  this->gicp_temp.setTransformationEpsilon(
      this->gicp_transformation_ep_); //设置gicp的参数,这个值代表两次迭代之间的最小差异
  this->gicp_temp.setRotationEpsilon(
      this->gicp_rotation_ep_); //设置gicp的参数,这个值代表两次迭代之间的最小旋转差异
  this->gicp_temp.setInitialLambdaFactor(
      this->gicp_init_lambda_factor_); //设置gicp的参数,这个值代表初始lambda因子

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(
      temp, true); //设置gicp的参数,这个值代表搜索源点的方法
  this->gicp.setSearchMethodTarget(
      temp, true); //设置gicp的参数,这个值代表搜索目标点的方法
  this->gicp_temp.setSearchMethodSource(
      temp, true); //设置gicp的参数,这个值代表搜索源点的方法
  this->gicp_temp.setSearchMethodTarget(
      temp, true); //设置gicp的参数,这个值代表搜索目标点的方法

  this->geo.first_opt_done = false; //初始化几何观测的第一次优化
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.); //初始化几何观测的上一次速度

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true); //设置crop的参数,让所有内部的点都被删除
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_,
                                    -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_,
                                    this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_,
                          this->vf_res_); //设置voxel的参数,这个值代表体素的大小

  this->metrics.spaciousness.push_back(0.); //初始化度量指标
  this->metrics.density.push_back(this->gicp_max_corr_dist_); //初始化度量指标

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

#ifdef HAS_CPUID //如果有cpuid
  unsigned int CPUInfo[4] = {0, 0, 0, 0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2],
          CPUInfo[3]); //获取CPU的信息
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002) //获取CPU的型号
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
#endif

  FILE *file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);       //获取CPU的时间
  this->lastSysCPU = timeSample.tms_stime;  //获取CPU的系统时间
  this->lastUserCPU = timeSample.tms_utime; //获取CPU的用户时间

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "processor", 9) == 0)
      this->numProcessors++;
  }
  fclose(file);
}

dlio::OdomNode::~OdomNode() {}

void dlio::OdomNode::getParams() {

  // Version
  ros::param::param<std::string>("~dlio/version", this->version_, "0.0.0");

  // Frames
  ros::param::param<std::string>("~dlio/frames/odom", this->odom_frame, "odom");
  ros::param::param<std::string>("~dlio/frames/baselink", this->baselink_frame,
                                 "base_link");
  ros::param::param<std::string>("~dlio/frames/lidar", this->lidar_frame,
                                 "lidar");
  ros::param::param<std::string>("~dlio/frames/imu", this->imu_frame, "imu");

  // Get Node NS and Remove Leading Character
  std::string ns = ros::this_node::getNamespace();
  ns.erase(0, 1);

  // Concatenate Frame Name Strings
  this->odom_frame = ns + "/" + this->odom_frame;
  this->baselink_frame = ns + "/" + this->baselink_frame;
  this->lidar_frame = ns + "/" + this->lidar_frame;
  this->imu_frame = ns + "/" + this->imu_frame;

  // Deskew FLag
  ros::param::param<bool>("~dlio/pointcloud/deskew", this->deskew_, true);

  // Gravity
  ros::param::param<double>("~dlio/odom/gravity", this->gravity_, 9.80665);

  // Keyframe Threshold
  ros::param::param<double>("~dlio/odom/keyframe/threshD",
                            this->keyframe_thresh_dist_, 0.1);
  ros::param::param<double>("~dlio/odom/keyframe/threshR",
                            this->keyframe_thresh_rot_, 1.0);

  // Submap
  ros::param::param<int>("~dlio/odom/submap/keyframe/knn", this->submap_knn_,
                         10);
  ros::param::param<int>("~dlio/odom/submap/keyframe/kcv", this->submap_kcv_,
                         10);
  ros::param::param<int>("~dlio/odom/submap/keyframe/kcc", this->submap_kcc_,
                         10);

  // Dense map resolution
  ros::param::param<bool>("~dlio/map/dense/filtered", this->densemap_filtered_,
                          true);

  // Wait until movement to publish map
  ros::param::param<bool>("~dlio/map/waitUntilMove", this->wait_until_move_,
                          false);

  // Crop Box Filter
  ros::param::param<double>("~dlio/odom/preprocessing/cropBoxFilter/size",
                            this->crop_size_, 1.0);

  // Voxel Grid Filter
  ros::param::param<bool>("~dlio/pointcloud/voxelize", this->vf_use_, true);
  ros::param::param<double>("~dlio/odom/preprocessing/voxelFilter/res",
                            this->vf_res_, 0.05);

  // Adaptive Parameters
  ros::param::param<bool>("~dlio/adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<float> t_default{0., 0., 0.};
  std::vector<float> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<float> baselink2imu_t, baselink2imu_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/t",
                                        baselink2imu_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/R",
                                        baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
      Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
      Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
          baselink2imu_R.data(), 3, 3);

  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) =
      this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) =
      this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<float> baselink2lidar_t, baselink2lidar_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/t",
                                        baselink2lidar_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/R",
                                        baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t = Eigen::Vector3f(
      baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
      Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
          baselink2lidar_R.data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) =
      this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) =
      this->extrinsics.baselink2lidar.R;

  // IMU
  ros::param::param<bool>("~dlio/odom/imu/calibration/accel",
                          this->calibrate_accel_, true);
  ros::param::param<bool>("~dlio/odom/imu/calibration/gyro",
                          this->calibrate_gyro_, true);
  ros::param::param<double>("~dlio/odom/imu/calibration/time",
                            this->imu_calib_time_, 3.0);
  ros::param::param<int>("~dlio/odom/imu/bufferSize", this->imu_buffer_size_,
                         2000);

  std::vector<float> accel_default{0., 0., 0.};
  std::vector<float> prior_accel_bias;
  std::vector<float> gyro_default{0., 0., 0.};
  std::vector<float> prior_gyro_bias;

  ros::param::param<bool>("~dlio/odom/imu/approximateGravity",
                          this->gravity_align_, true);
  ros::param::param<bool>("~dlio/imu/calibration", this->imu_calibrate_, true);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/bias",
                                        prior_accel_bias, accel_default);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/gyro/bias",
                                        prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<float> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<float> imu_sm;

  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/sm", imu_sm,
                                        imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ =
        Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
            imu_sm.data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  ros::param::param<int>("~dlio/odom/gicp/minNumPoints",
                         this->gicp_min_num_points_, 100);
  ros::param::param<int>("~dlio/odom/gicp/kCorrespondences",
                         this->gicp_k_correspondences_, 20);
  ros::param::param<double>("~dlio/odom/gicp/maxCorrespondenceDistance",
                            this->gicp_max_corr_dist_,
                            std::sqrt(std::numeric_limits<double>::max()));
  ros::param::param<int>("~dlio/odom/gicp/maxIterations", this->gicp_max_iter_,
                         64);
  ros::param::param<double>("~dlio/odom/gicp/transformationEpsilon",
                            this->gicp_transformation_ep_, 0.0005);
  ros::param::param<double>("~dlio/odom/gicp/rotationEpsilon",
                            this->gicp_rotation_ep_, 0.0005);
  ros::param::param<double>("~dlio/odom/gicp/initLambdaFactor",
                            this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  ros::param::param<double>("~dlio/odom/geo/Kp", this->geo_Kp_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kv", this->geo_Kv_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kq", this->geo_Kq_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kab", this->geo_Kab_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kgb", this->geo_Kgb_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/abias_max", this->geo_abias_max_,
                            1.0);
  ros::param::param<double>("~dlio/odom/geo/gbias_max", this->geo_gbias_max_,
                            1.0);
}

void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout
      << std::endl
      << "+-------------------------------------------------------------------+"
      << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v"
            << this->version_ << "               |" << std::endl;
  std::cout
      << "+-------------------------------------------------------------------+"
      << std::endl;
}

void dlio::OdomNode::publishPose(const ros::TimerEvent &e) {

  // nav_msgs::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub.publish(this->odom_ros);

  // geometry_msgs::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub.publish(this->pose_ros); //发送位姿信息
}

void dlio::OdomNode::publishToROS(
    pcl::PointCloud<PointType>::ConstPtr published_cloud,
    Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub.publish(this->path_ros);

  // transform: odom to baselink
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br.sendTransform(transformStamped); //发布tf

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x =
      this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y =
      this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z =
      this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br.sendTransform(transformStamped);
}

void dlio::OdomNode::publishCloud(
    pcl::PointCloud<PointType>::ConstPtr published_cloud,
    Eigen::Matrix4f T_cloud) {

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) {
      return;
    }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_(
      boost::make_shared<pcl::PointCloud<PointType>>()); //创建一个点云指针

  pcl::transformPointCloud(*published_cloud, *deskewed_scan_t_, T_cloud);

  // 发布去畸变的点云
  sensor_msgs::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  deskewed_ros.header.stamp = this->scan_header_stamp;
  deskewed_ros.header.frame_id = this->odom_frame;
  this->deskewed_pub.publish(deskewed_ros);
}

void dlio::OdomNode::publishKeyframe(
    std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
              pcl::PointCloud<PointType>::ConstPtr>
        kf,
    ros::Time timestamp) {

  // Push back
  geometry_msgs::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub.publish(this->kf_pose_ros);

  // 发布地图的关键帧扫描
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub.publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub.publish(keyframe_cloud_ros);
  }
}

void dlio::OdomNode::getScanFromROS(
    const sensor_msgs::PointCloud2ConstPtr &pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_(
      boost::make_shared<pcl::PointCloud<PointType>>());
  pcl::fromROSMsg(*pc, *original_scan_);

  // 去除无效点
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // crop框选范围内的点
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // 自动检测传感器类型
  this->sensor = dlio::SensorType::UNKNOWN;
  for (auto &field : pc->fields) {
    if (field.name == "t") {
      this->sensor = dlio::SensorType::OUSTER;
      break;
    } else if (field.name == "time") {
      this->sensor = dlio::SensorType::VELODYNE; // velodyne雷达
      break;
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;
}

void dlio::OdomNode::preprocessPoints() {

  // 取消原始数据类型扫描
  if (this->deskew_) { //如果可以去畸变。也就是不为dlio::SensorType::UNKNOWN

    this->deskewPointcloud(); //去畸变

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = this->scan_header_stamp.toSec();

    // 在IMU数据到达之前不要处理扫描数据
    if (!this->first_valid_scan) { // tips：这个和下面的deskewPointcloud函数中判断类似

      if (this->imu_buffer.empty() ||
          this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // 假设第一次扫描没有运动

    } else {

      // 第二个及以后的扫描的IMU先验
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
          frames;
      frames = this->integrateImu(
          this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
          this->geo.prev_vel.cast<float>(), {this->scan_stamp}); // IMU积分

      if (frames.size() > 0) {
        this->T_prior = frames.back();
      } else {
        this->T_prior = this->T;
      }
    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_(
        boost::make_shared<pcl::PointCloud<PointType>>()); //创建一个点云指针
    pcl::transformPointCloud(
        *this->original_scan, *deskewed_scan_,
        this->T_prior *
            this->extrinsics
                .baselink2lidar_T); //将原始点云转换到baselink坐标系下
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_(
        boost::make_shared<pcl::PointCloud<PointType>>(
            *this->deskewed_scan));           //创建一个点云指针
    this->voxel.setInputCloud(current_scan_); //设置输入点云
    this->voxel.filter(*current_scan_);       //滤波
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }
}

void dlio::OdomNode::deskewPointcloud() {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_(
      boost::make_shared<pcl::PointCloud<PointType>>());
  deskewed_scan_->points.resize(
      this->original_scan->points.size()); //设置点云大小

  // 各个点的时间戳应该相对于此时间
  double sweep_ref_time = this->scan_header_stamp.toSec();

  // 按时间戳对点进行排序并构建时间戳列表
  std::function<bool(const PointType &, const PointType &)>
      point_time_cmp; //比较函数
  std::function<bool(boost::range::index_value<PointType &, long>,
                     boost::range::index_value<PointType &, long>)>
      point_time_neq; //不等于函数
  std::function<double(boost::range::index_value<PointType &, long>)>
      extract_point_time; //提取时间

  if (this->sensor == dlio::SensorType::OUSTER) {
    point_time_cmp = [](const PointType &p1, const PointType &p2) {
      return p1.t < p2.t;
    }; //定义内容
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2) {
      return p1.value().t != p2.value().t;
    };
    extract_point_time =
        [&sweep_ref_time](boost::range::index_value<PointType &, long> pt) {
          return sweep_ref_time + pt.value().t * 1e-9f;
        };
  } else {
    point_time_cmp = [](const PointType &p1, const PointType &p2) {
      return p1.time < p2.time;
    };
    point_time_neq = [](boost::range::index_value<PointType &, long> p1,
                        boost::range::index_value<PointType &, long> p2) {
      return p1.value().time != p2.value().time;
    };
    extract_point_time =
        [&sweep_ref_time](boost::range::index_value<PointType &, long> pt) {
          return sweep_ref_time + pt.value().time;
        };
  }

  // 按照时间戳的顺序将点复制到deskewed_scan_中
  std::partial_sort_copy(this->original_scan->points.begin(),
                         this->original_scan->points.end(),
                         deskewed_scan_->points.begin(),
                         deskewed_scan_->points.end(), point_time_cmp);

  // 这个函数的作用是从deskewed_scan_点云数据中提取出时间戳不相同的点，并将它们存储在一个名为points_unique_timestamps的变量中。
  // 这个函数使用了Boost库中的adaptors，首先使用indexed()将点序号和点本身组合成一个pair，然后使用adjacent_filtered()过滤掉时间戳相邻的相同点，最终得到时间戳不相同的点。
  auto points_unique_timestamps =
      deskewed_scan_->points | boost::adaptors::indexed() |
      boost::adaptors::adjacent_filtered(point_time_neq);

  // 从点中提取时间戳并将它们放入一个独立的列表中
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;
  for (auto it = points_unique_timestamps.begin();
       it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it));
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(
      deskewed_scan_->points.size()); //最后一个存点的个数

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp =
      timestamps[median_pt_index]; // 将this->scan_stamp设置为中位点的时间戳

  //在IMU数据到达之前不要处理扫描数据
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() ||
        this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // 假设第一次扫描时没有运动
    pcl::transformPointCloud(*deskewed_scan_, *deskewed_scan_,
                             this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  //从第二次扫描开始，使用IMU先验和去斜校正
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
      frames;
  frames = this->integrateImu(
      this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
      this->geo.prev_vel.cast<float>(),
      timestamps); // IMU积分，传入timestamps这个vector来获取每个点时间的IMU积分
  this->deskew_size =
      frames.size(); // 如果积分成功，则时间戳的数量应该等于timestamps.size()

  // 如果扫描开始和结束之间没有帧，则可能意味着存在同步问题
  if (frames.size() != timestamps.size()) {
    ROS_FATAL("Bad time sync between LiDAR and IMU!");

    this->T_prior = this->T; //那直接将T_prior设置为T
    pcl::transformPointCloud(
        *deskewed_scan_, *deskewed_scan_,
        this->T_prior *
            this->extrinsics
                .baselink2lidar_T); //将原始点云转换到baselink坐标系下
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // 将先验更新为扫描中间时间的估计姿态（对应于this->scan_stamp）
  this->T_prior = frames[median_pt_index];

#pragma omp parallel for num_threads(this->num_threads_) //并行计算
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T =
        frames[i] * this->extrinsics.baselink2lidar_T; //设置变换矩阵

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i + 1]; k++) {
      auto &pt = deskewed_scan_->points[k]; //取出点云中的点
      pt.getVector4fMap()[3] = 1.; //将点云中的点转换为齐次坐标
      pt.getVector4fMap() =
          T * pt.getVector4fMap(); //将点云中的点转换到baselink坐标系下
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;
}

void dlio::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // 保留关键帧的历史记录
  this->keyframes.push_back(
      std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q),
                     this->current_scan)); //向keyframes记录位置和扫描的点云
  this->keyframe_timestamps.push_back(
      this->scan_header_stamp); //向keyframe_timestamps记录时间
  this->keyframe_normals.push_back(
      this->gicp.getSourceCovariances()); //向keyframe_normals记录协防差
  this->keyframe_transformations.push_back(
      this->T_corr); //向keyframe_transformations记录上一帧和这一帧的位移量
}

void dlio::OdomNode::setInputSource() {
  this->gicp.setInputSource(
      this->current_scan); //将当前帧设置为输入源,传入到GICP中
  this->gicp.calculateSourceCovariances(); //计算输入源的协方差矩阵
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received ||
      !this->imu_calibrated) { //如果没有接收到IMU数据或者IMU没有校准
    return;
  }

  this->dlio_initialized = true; //初始化完成
  std::cout << std::endl << " DLIO initialized!" << std::endl;
}

void dlio::OdomNode::callbackPointCloud(
    const sensor_msgs::PointCloud2ConstPtr &pc) {

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(
      main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  double then = ros::Time::now().toSec();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = pc->header.stamp.toSec();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // 将传入的扫描转换为DLIO格式
  this->getScanFromROS(pc);

  // 预处理点云
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    ROS_FATAL("Low number of points in the cloud!");
    return;
  }

  // 计算度量指标
  this->metrics_thread = std::thread(&dlio::OdomNode::computeMetrics, this);
  this->metrics_thread.detach();

  // 设置自适应参数
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // 将新帧设置为输入源，并传入GICP
  this->setInputSource();

  // 将初始帧设置为第一关键帧
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running =
        false; //将main_loop_running设置为false,告诉buildKeyframesAndSubmap可以开始创建子图了
    this->submap_future = std::async(
        std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this,
        this->state); //调用buildKeyframesAndSubmap完成子图创建,第一次必定创建子图
    this->submap_future.wait(); // 等待任务完成
    return;
  }

  // 通过IMU + S2M + GEO获取下一个姿态
  this->getNextPose();

  // 更新当前关键帧姿态和地图
  this->updateKeyframes();

  // 如果需要，构建关键帧法线和子地图（如果我们还没有在等待中）
  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future =
        std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap,
                   this, this->state);
  } else {
    lock.lock();
    this->main_loop_running =
        false; //将main_loop_running设置为false,告诉buildKeyframesAndSubmap可以开始创建子图了
    lock.unlock();
    this->submap_build_cv
        .notify_one(); //通知buildKeyframesAndSubmap可以开始创建子图了
  }

  // 更新轨迹
  this->trajectory.push_back(std::make_pair(this->state.p, this->state.q));

  // 更新时间戳
  this->lidar_rates.push_back(1. / (this->scan_stamp - this->prev_scan_stamp));
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // 将信息发布到ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  this->publish_thread = std::thread(&dlio::OdomNode::publishToROS, this,
                                     published_cloud, this->T_corr);
  this->publish_thread.detach();

  // 更新一些统计数据
  this->comp_times.push_back(ros::Time::now().toSec() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // 调试语句和发布自定义DLIO消息
  this->debug_thread = std::thread(&dlio::OdomNode::debug, this);
  this->debug_thread.detach();

  this->geo.first_opt_done = true; //第一次优化完成
}

void dlio::OdomNode::callbackImu(const sensor_msgs::Imu::ConstPtr &imu_raw) {

  this->first_imu_received = true; //接收到IMU数据

  sensor_msgs::Imu::Ptr imu = this->transformImu(imu_raw); //转换IMU数据
  this->imu_stamp = imu->header.stamp;

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // 获取IMU信息
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu->header.stamp.toSec(); //第一次IMU时间戳
  }

  // IMU校准程序 - 进行三秒钟
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg(0., 0., 0.);
    static Eigen::Vector3f accel_avg(0., 0., 0.);
    static bool print = true;

    if ((imu->header.stamp.toSec() - this->first_imu_stamp) <
        this->imu_calib_time_) { //如果时间小于3s

      num_samples++; //计数

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if (print) {
        std::cout << std::endl
                  << " Calibrating IMU for " << this->imu_calib_time_
                  << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples; //计算平均值
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec(0., 0., this->gravity_); //重力向量

      if (this->gravity_align_) { //如果需要重力校准

        // 估计重力向量 - 如果偏差未经预校准，则仅为近似值
        grav_vec = (accel_avg - this->state.b.accel).normalized() *
                   abs(this->gravity_); //重力向量，根据真实的ba来计算
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(
            grav_vec,
            Eigen::Vector3f(
                0., 0.,
                this->gravity_)); //重力向量对应的四元数，通过FromTwoVectors求出两个的夹角

        // 设置重力对齐方向
        this->state.q = grav_q;
        this->T.block(0, 0, 3, 3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0 / M_PI);
        double pitch = euler[1] * (180.0 / M_PI);
        double roll = euler[2] * (180.0 / M_PI);

        // 如果偏航角较小，请使用备用表示
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw = remainder(yaw + 180.0, 360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll = remainder(roll + 180.0, 360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4)
                  << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4)
                  << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4)
                  << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) { //如果需要校准加速度计

        // 将重力从平均加速度中减去以得到偏差
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: "
                  << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                  << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                  << to_string_with_precision(this->state.b.accel[2], 8)
                  << std::endl;
      }

      if (this->calibrate_gyro_) { //如果需要校准陀螺仪

        this->state.b.gyro = gyro_avg; //计算陀螺仪的偏差

        std::cout << " Gyro biases  [xyz]: "
                  << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                  << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                  << to_string_with_precision(this->state.b.gyro[2], 8)
                  << std::endl;
      }

      this->imu_calibrated = true; // IMU校准完成
    }

  } else {

    double dt = imu->header.stamp.toSec() - this->prev_imu_stamp; //计算时间差
    this->imu_rates.push_back(1. / dt);
    if (dt == 0) {
      return;
    }

    // 将校准偏差应用于新的IMU测量数据
    this->imu_meas.stamp = imu->header.stamp.toSec();
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected =
        (this->imu_accel_sm_ * lin_accel) - this->state.b.accel; //加速度计校准
    Eigen::Vector3f ang_vel_corrected =
        ang_vel - this->state.b.gyro; //陀螺仪校准

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // 将校准后的IMU测量值存储到IMU缓冲区中，以备后续手动集成
    this->mtx_imu.lock();
    this->imu_buffer.push_front(
        this->imu_meas); //将IMU数据存储到缓冲区中,从前面插入
    this->mtx_imu.unlock();

    // 通知callbackPointCloud线程，当前时间存在IMU数据
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // 几何观察器：传播状态
      this->propagateState();
    }
  }
}

/**
 * @brief 通过IMU + S2M + GEO获取下一个姿态
 *
 */
void dlio::OdomNode::getNextPose() {

  // 检查新子地图是否准备好可供使用
  this->new_submap_is_ready =
      (this->submap_future.wait_for(std::chrono::seconds(0)) ==
       std::future_status::ready); //等待子地图准备好

  if (this->new_submap_is_ready &&
      this->submap_hasChanged) { //如果子地图准备好了,并且子地图发生了变化

    // 将当前全局子地图设置为目标点云
    this->gicp.registerInputTarget(this->submap_cloud);

    // 设置子图的kdtree,之前就是直接从target_kdtree_拿出来的,这个有必要嘛?
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // 将目标云的法线设置为子地图的法线
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  // 使用全局IMU变换作为初始猜测，将当前子地图与全局地图对齐
  pcl::PointCloud<PointType>::Ptr aligned(
      boost::make_shared<pcl::PointCloud<PointType>>());
  this->gicp.align(*aligned); // 设置对齐后的地图

  // 在全局坐标系中获取最终变换
  this->T_corr =
      this->gicp.getFinalTransformation(); // 根据对齐后的地图来校准转换
  this->T = this->T_corr * this->T_prior;

  // 更新下一个全局位姿,现在源点云和目标点云都在全局坐标系中，所以变换是全局的
  this->propagateGICP();

  // 几何观察器更新
  this->updateState();
}

/**
 * @brief 根据时间范围获取IMU测量值
 *
 * @param start_time 需要的IMU测量值的开始时间
 * @param end_time 需要的IMU测量值的结束时间
 * @param begin_imu_it 对应的imu_buffer的起始索引
 * @param end_imu_it 对应的imu_buffer的结束索引
 * @return true
 * @return false
 */
bool dlio::OdomNode::imuMeasFromTimeRange(
    double start_time, double end_time,
    boost::circular_buffer<ImuMeas>::reverse_iterator &begin_imu_it,
    boost::circular_buffer<ImuMeas>::reverse_iterator &end_imu_it) {

  //如果IMU缓冲区为空或者IMU缓冲区的第一个IMU数据的时间戳小于end_time
  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // 等待最新的IMU数据
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu); //互斥锁
    this->cv_imu_stamp.wait(lock, [this, &end_time] {
      return this->imu_buffer.front().stamp >= end_time; //等待最新的IMU数据
    }); //等待最新的IMU数据
  }

  auto imu_it =
      this->imu_buffer.begin(); // IMU数据的迭代器，直到IMU数据要在end_time之后

  auto last_imu_it = imu_it; //设置最新的imu作为最后的imu
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it; //不断迭代拿到之前时间的IMU数据
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++; //不断迭代拿到之前时间的IMU数据,并拿到imu_it和last_imu_it之间的IMU数据
  }

  if (imu_it == this->imu_buffer.end()) {
    // 测量不足，返回false，因为没有值在end_time之后
    return false;
  }
  imu_it++;

  // 设置反向迭代器（以正向时间迭代）
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

/**
 * @brief
 * 这个函数主要是用来处理IMU数据的，主要是对IMU数据进行积分，得到每个点的IMU积分
 *
 * @param start_time 上一帧扫描的时间戳
 * @param q_init 上一帧估算的姿态
 * @param p_init 上一帧估算的位置
 * @param v_init 上一帧估算的速度
 * @param sorted_timestamps 当前帧对应点(帧)扫描的时间戳
 * @return std::vector<Eigen::Matrix4f,
 * Eigen::aligned_allocator<Eigen::Matrix4f>>
 */
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init,
                             Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                             const std::vector<double> &sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
      empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // 如果时间戳为空或者时间戳大于第一个时间戳，认为无效的输入，返回空向量
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(),
                                 begin_imu_it, end_imu_it) == false) {
    //从IMU数据中提取出时间戳不相同的点，IMU测量不足，返回空向量。
    return empty;
  }

  // 反向整合以找到第一个IMU样本的姿态
  const ImuMeas &f1 = *begin_imu_it;
  const ImuMeas &f2 = *(begin_imu_it + 1);

  // 两个IMU样本之间的时间
  double dt = f2.dt;

  // 第一个IMU样本和开始时间之间的时间间隔
  double idt = start_time - f1.stamp;

  //第一和第二个IMU样本之间的角加速度
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // 首个IMU样本和起始时间之间的平均角速度（反向）
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5 * alpha * idt);

  // 将q_init设置为第一个IMU样本的方向
  q_init = Eigen::Quaternionf(
      q_init.w() - 0.5 *
                       (q_init.x() * omega_i[0] + q_init.y() * omega_i[1] +
                        q_init.z() * omega_i[2]) *
                       idt,
      q_init.x() + 0.5 *
                       (q_init.w() * omega_i[0] - q_init.z() * omega_i[1] +
                        q_init.y() * omega_i[2]) *
                       idt,
      q_init.y() + 0.5 *
                       (q_init.z() * omega_i[0] + q_init.w() * omega_i[1] -
                        q_init.x() * omega_i[2]) *
                       idt,
      q_init.z() +
          0.5 *
              (q_init.x() * omega_i[1] - q_init.y() * omega_i[0] +
               q_init.w() * omega_i[2]) *
              idt); //根据第一个IMU推算出开始时间，对应公式4克罗内克积，省略了后面一项
  q_init.normalize();

  // 第一和第二个IMU样本之间的平均角速度
  Eigen::Vector3f omega = f1.ang_vel + 0.5 * alpha_dt;

  // 第二个惯性测量单元样本的方向
  Eigen::Quaternionf q2(
      q_init.w() - 0.5 *
                       (q_init.x() * omega[0] + q_init.y() * omega[1] +
                        q_init.z() * omega[2]) *
                       dt,
      q_init.x() + 0.5 *
                       (q_init.w() * omega[0] - q_init.z() * omega[1] +
                        q_init.y() * omega[2]) *
                       dt,
      q_init.y() + 0.5 *
                       (q_init.z() * omega[0] + q_init.w() * omega[1] -
                        q_init.x() * omega[2]) *
                       dt,
      q_init.z() + 0.5 *
                       (q_init.x() * omega[1] - q_init.y() * omega[0] +
                        q_init.w() * omega[2]) *
                       dt);
  q2.normalize();

  // 首个IMU样本的加速度
  Eigen::Vector3f a1 =
      q_init._transformVector(f1.lin_accel); //根据线加速度求出f1时刻的加速度
  a1[2] -= this->gravity_;                   //减去重力加速度

  // 第二个IMU样本的加速度
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // 在前两个IMU采样之间的jerk
  Eigen::Vector3f j = (a2 - a1) / dt;

  // 将 v_init 设置为第一个IMU样本的速度（从 start_time 开始向后倒退），公式4
  v_init -= a1 * idt + 0.5 * j * idt * idt;

  //将p_init设置为第一个IMU样本的位置（从start_time往回走，公式4
  p_init -=
      v_init * idt + 0.5 * a1 * idt * idt + (1 / 6.) * j * idt * idt * idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps,
                                    begin_imu_it, end_imu_it);
}

/**
 * @brief
 * 这里可以根据所有的点以及IMU时间完成计算，并对其进行插值，以便在任何时间点都可以获得姿态。对应公式5
 *
 * @param q_init 计算得到的初始姿态
 * @param p_init 计算得到的初始位置
 * @param v_init 计算得到的初始速度
 * @param sorted_timestamps 所有点的时间戳
 * @param begin_imu_it 第一个IMU样本的迭代器
 * @param end_imu_it  最后一个IMU样本的迭代器
 * @return std::vector<Eigen::Matrix4f,
 * Eigen::aligned_allocator<Eigen::Matrix4f>>
 */
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(
    Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
    const std::vector<double> &sorted_timestamps,
    boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
    boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
      imu_se3;

  // 初始化
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // 遍历IMU测量值和时间戳
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it =
      sorted_timestamps.begin(); // 对应的所有点的索引，并将起始值放入

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas &f0 = *prev_imu_it;
    const ImuMeas &f = *imu_it;

    // IMU样本之间的时间
    double dt = f.dt;

    // 角加速度
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // 平均角速度
    Eigen::Vector3f omega = f0.ang_vel + 0.5 * alpha_dt;

    // 根据公式4计算的四元数，因为只传入了一个IMU样本，所以这里的dt就是IMU样本之间的时间
    q = Eigen::Quaternionf(
        q.w() -
            0.5 * (q.x() * omega[0] + q.y() * omega[1] + q.z() * omega[2]) * dt,
        q.x() +
            0.5 * (q.w() * omega[0] - q.z() * omega[1] + q.y() * omega[2]) * dt,
        q.y() +
            0.5 * (q.z() * omega[0] + q.w() * omega[1] - q.x() * omega[2]) * dt,
        q.z() + 0.5 * (q.x() * omega[1] - q.y() * omega[0] + q.w() * omega[2]) *
                    dt);
    q.normalize();

    // 加速度
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // 为给定的时间戳进行插值
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // 上一个IMU采样点和给定时间戳之间的时间间隔
      double idt = *stamp_it - f0.stamp;

      // 平均角速度
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5 * alpha * idt;

      // 根据公式5计算的四元数
      Eigen::Quaternionf q_i(
          q.w() - 0.5 *
                      (q.x() * omega_i[0] + q.y() * omega_i[1] +
                       q.z() * omega_i[2]) *
                      idt,
          q.x() + 0.5 *
                      (q.w() * omega_i[0] - q.z() * omega_i[1] +
                       q.y() * omega_i[2]) *
                      idt,
          q.y() + 0.5 *
                      (q.z() * omega_i[0] + q.w() * omega_i[1] -
                       q.x() * omega_i[2]) *
                      idt,
          q.z() + 0.5 *
                      (q.x() * omega_i[1] - q.y() * omega_i[0] +
                       q.w() * omega_i[2]) *
                      idt);
      q_i.normalize();

      // 根据公式5计算的位置
      Eigen::Vector3f p_i =
          p + v * idt + 0.5 * a0 * idt * idt + (1 / 6.) * j * idt * idt * idt;

      // 以matrix矩阵形式表示
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v * dt + 0.5 * a0 * dt * dt + (1 / 6.) * j_dt * dt * dt;

    // Velocity
    v += a0 * dt + 0.5 * j_dt * dt;

    prev_imu_it = imu_it;
  }

  return imu_se3;
}

/**
 * @brief 从GICP中拿到结果并更新
 *
 */
void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0, 3), this->T(1, 3), this->T(2, 3); // 位置

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0, 0), this->T(0, 1), this->T(0, 2), this->T(1, 0),
      this->T(1, 1), this->T(1, 2), this->T(2, 0), this->T(2, 1),
      this->T(2, 2); // 旋转矩阵

  Eigen::Quaternionf q(rotSO3);

  // 对四元数求范数
  double norm =
      sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  q.w() /= norm;
  q.x() /= norm;
  q.y() /= norm;
  q.z() /= norm;
  this->lidarPose.q = q;
}

void dlio::OdomNode::propagateState() {

  // 锁定线程以防止 UpdateState 访问状态
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  double dt = this->imu_meas.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // 将加速度从机体坐标系转换到世界坐标系
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // 加速度传播
  this->state.p[0] +=
      this->state.v.lin.w[0] * dt + 0.5 * dt * dt * world_accel[0];
  this->state.p[1] +=
      this->state.v.lin.w[1] * dt + 0.5 * dt * dt * world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2] * dt +
                      0.5 * dt * dt * (world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0] * dt;
  this->state.v.lin.w[1] += world_accel[1] * dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_) * dt;
  this->state.v.lin.b =
      this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // 重力计传播
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // 确保四元数已经正确归一化
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
}

void dlio::OdomNode::updateState() {

  // 锁定线程以防止状态被PropagateState访问
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  Eigen::Vector3f pin = this->lidarPose.p;              // 位置
  Eigen::Quaternionf qin = this->lidarPose.q;           // 四元数
  double dt = this->scan_stamp - this->prev_scan_stamp; // 时间差

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // 构造误差的四元数
  qe = qhat.conjugate() * qin; //通过拿到的四元数和预测的四元数构造误差的四元数

  double sgn = 1.;
  if (qe.w() < 0) { // 如果误差的四元数的w小于0
    sgn = -1;
  }

  // 构建四元数校正量,对应公式7
  qcorr.w() = 1 - abs(qe.w());  // 误差的四元数的w部分
  qcorr.vec() = sgn * qe.vec(); // 误差的四元数的向量部分
  qcorr = qhat * qcorr;         // 误差的四元数

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body =
      qhat.conjugate()._transformVector(err); // 误差的四元数转换到body坐标系下

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // 更新加速度偏差
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel =
      this->state.b.accel.array().min(abias_max).max(-abias_max);

  // 更新陀螺仪偏差
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro =
      this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // 更新速度和位置
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // 存储前一个姿态、方向和速度
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;
}

sensor_msgs::Imu::Ptr
dlio::OdomNode::transformImu(const sensor_msgs::Imu::ConstPtr &imu_raw) {

  sensor_msgs::Imu::Ptr imu(new sensor_msgs::Imu);

  // Copy header
  imu->header = imu_raw->header;

  static double prev_stamp = imu->header.stamp.toSec();
  double dt = imu->header.stamp.toSec() - prev_stamp;
  prev_stamp = imu->header.stamp.toSec();

  if (dt == 0) {
    dt = 1.0 / 200.0;
  }

  // Transform angular velocity (will be the same on a rigid body, so just
  // rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to
  // translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg =
      lin_accel_cg +
      ((ang_vel_cg - ang_vel_cg_prev) / dt)
          .cross(-this->extrinsics.baselink2imu.t) +
      ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;
}

void dlio::OdomNode::computeMetrics() {
  this->computeSpaciousness(); //计算稀疏度
  this->computeDensity();      //计算密度
}

void dlio::OdomNode::computeSpaciousness() {

  // 计算点的范围
  std::vector<float> ds;

  for (int i = 0; i <= this->original_scan->points.size();
       i++) { //根据getScanFromROS拿到的原始点云数据
    float d = std::sqrt(
        pow(this->original_scan->points[i].x, 2) +
        pow(this->original_scan->points[i].y, 2)); //计算点到原点的距离
    ds.push_back(d);                               //将距离存入ds
  }

  // 求中值
  std::nth_element(
      ds.begin(), ds.begin() + ds.size() / 2,
      ds.end()); // 用于在一个序列中找到第k小的元素，其中k由第二个参数指定
  float median_curr = ds[ds.size() / 2];  //对应的中值的索引
  static float median_prev = median_curr; //存入到上一个时刻的中值
  float median_lpf =
      0.95 * median_prev + 0.05 * median_curr; //？算出来还是一个值
  median_prev = median_lpf;                    //同理

  // push
  this->metrics.spaciousness.push_back(median_lpf);
}

void dlio::OdomNode::computeDensity() {

  float density;

  if (!this->geo
           .first_opt_done) { //如果第一次优化未完成（没有完成GICP），则认为没有密度
    density = 0.;
  } else {
    density = this->gicp.source_density_; //将GICP累计的density传入
  }

  static float density_prev = density;
  float density_lpf = 0.95 * density_prev + 0.05 * density;
  density_prev = density_lpf;

  this->metrics.density.push_back(density_lpf);
}

void dlio::OdomNode::computeConvexHull() {

  // 至少需要4个凸包关键帧
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // 创建一个点云，在关键帧处放置点
  pcl::PointCloud<PointType>::Ptr cloud = pcl::PointCloud<PointType>::Ptr(
      boost::make_shared<pcl::PointCloud<PointType>>());

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0]; //关键帧的位置
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // 计算关键帧的凸包
  this->convex_hull.setInputCloud(cloud);

  // 获取凸包上关键帧的索引
  pcl::PointCloud<PointType>::Ptr convex_points =
      pcl::PointCloud<PointType>::Ptr(
          boost::make_shared<pcl::PointCloud<PointType>>());
  this->convex_hull.reconstruct(
      *convex_points); // 通过传入的点集来重新构建当前对象中的凸包

  pcl::PointIndices::Ptr convex_hull_point_idx =
      pcl::PointIndices::Ptr(boost::make_shared<pcl::PointIndices>());
  this->convex_hull.getHullPointIndices(
      *convex_hull_point_idx); //获取凸包上的点的索引,这个索引是在cloud中的索引

  this->keyframe_convex.clear();
  for (int i = 0; i < convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(
        convex_hull_point_idx->indices[i]); //然后压入到keyframe_convex
  }
}

void dlio::OdomNode::computeConcaveHull() {

  // 至少需要4个凹包关键帧
  if (this->num_processed_keyframes < 5) {
    return;
  }

  //  创建一个点云，在关键帧处放置点
  pcl::PointCloud<PointType>::Ptr cloud = pcl::PointCloud<PointType>::Ptr(
      boost::make_shared<pcl::PointCloud<PointType>>());

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0]; //关键帧的位置
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // 计算关键帧的凹包
  this->concave_hull.setInputCloud(cloud);

  // 获取凹包上关键帧的索引
  pcl::PointCloud<PointType>::Ptr concave_points =
      pcl::PointCloud<PointType>::Ptr(
          boost::make_shared<pcl::PointCloud<PointType>>());
  this->concave_hull.reconstruct(
      *concave_points); // 通过传入的点集来重新构建当前对象中的凹包

  pcl::PointIndices::Ptr concave_hull_point_idx =
      pcl::PointIndices::Ptr(boost::make_shared<pcl::PointIndices>());
  this->concave_hull.getHullPointIndices(
      *concave_hull_point_idx); //获取凹包上的点的索引,这个索引是在cloud中的索引

  this->keyframe_concave.clear();
  for (int i = 0; i < concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(
        concave_hull_point_idx->indices[i]); //然后压入到keyframe_concave
  }
}

void dlio::OdomNode::updateKeyframes() {

  // 计算轨迹中所有姿态和旋转的差异
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto &k : this->keyframes) {

    // 计算当前姿态与关键帧中的姿态之间的距离,这里和更新submap的操作一样
    float delta_d = sqrt(pow(this->state.p[0] - k.first.first[0], 2) +
                         pow(this->state.p[1] - k.first.first[1], 2) +
                         pow(this->state.p[2] - k.first.first[2], 2));

    //计算当前姿态附近的数量
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5) {
      ++num_nearby;
    }

    // 将其存储到变量中
    if (delta_d < closest_d) {
      closest_d = delta_d;         //最近的距离
      closest_idx = keyframes_idx; //最近的关键帧的索引
    }

    keyframes_idx++;
  }

  // 获取最接近的姿势和相应的旋转
  Eigen::Vector3f closest_pose =
      this->keyframes[closest_idx].first.first; //最近的关键帧的位置
  Eigen::Quaternionf closest_pose_r =
      this->keyframes[closest_idx].first.second; //最近的关键帧的旋转

  // 计算当前姿势与最近的姿势之间的距离,和closest_d一致
  float dd = sqrt(pow(this->state.p[0] - closest_pose[0], 2) +
                  pow(this->state.p[1] - closest_pose[1], 2) +
                  pow(this->state.p[2] - closest_pose[2], 2));

  // 使用SLERP计算方向差异
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) <
      0.) { //如果两个四元数的点积小于0，说明两个四元数的方向相反
    Eigen::Quaternionf lq = closest_pose_r; //将最近的关键帧的旋转赋值给lq
    lq.w() *= -1.;
    lq.x() *= -1.;
    lq.y() *= -1.;
    lq.z() *= -1.;
    dq = this->state.q * lq.inverse(); //计算当前姿态与最近的姿态之间的旋转
  } else {
    dq = this->state.q *
         closest_pose_r.inverse(); //计算当前姿态与最近的姿态之间的旋转
  }

  double theta_rad =
      2. * atan2(sqrt(pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2)),
                 dq.w()); //计算当前姿态与最近的姿态之间的旋转角度
  double theta_deg = theta_rad * (180.0 / M_PI); //将弧度转换为角度

  // 更新关键帧
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ ||
      abs(theta_deg) >
          this->keyframe_thresh_rot_) { //如果距离或者旋转角度超过阈值
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ &&
      abs(theta_deg) > this->keyframe_thresh_rot_ &&
      num_nearby <=
          1) { //如果距离小于阈值，但是旋转角度超过阈值，且附近的关键帧数量小于等于1
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) { //如果距离小于阈值
    newKeyframe = false;
  } else if (abs(dd) <= 0.5) {
    newKeyframe = false;
  }

  if (newKeyframe) {

    // 更新关键帧向量
    std::unique_lock<decltype(this->keyframes_mutex)> lock(
        this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(
        std::make_pair(this->lidarPose.p, this->lidarPose.q),
        this->current_scan)); //将当前的姿态和点云压入到keyframes中
    this->keyframe_timestamps.push_back(
        this->scan_header_stamp); //将当前的时间戳压入到keyframe_timestamps中
    this->keyframe_normals.push_back(
        this->gicp
            .getSourceCovariances()); //将当前的法向量压入到keyframe_normals中
    this->keyframe_transformations.push_back(
        this->T_corr); //将当前的变换矩阵压入到keyframe_transformations中
    lock.unlock();
  }
}

void dlio::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back(); // 设置从metrics拿到的宽敞度

  //设置最大最小值
  if (sp < 0.5) {
    sp = 0.5;
  }
  if (sp > 5.0) {
    sp = 5.0;
  }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back(); //拿到最新的稠密度

  if (den < 0.5 * this->gicp_max_corr_dist_) {
    den = 0.5 * this->gicp_max_corr_dist_;
  }
  if (den > 2.0 * this->gicp_max_corr_dist_) {
    den = 2.0 * this->gicp_max_corr_dist_;
  }

  if (sp < 5.0) {
    den = 0.5 * this->gicp_max_corr_dist_;
  };
  if (sp > 5.0) {
    den = 2.0 * this->gicp_max_corr_dist_;
  };

  this->gicp.setMaxCorrespondenceDistance(den); //设置GICP中最大配准距离

  // 凹包的值
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
}

/**
 * @brief 压入子图索引
 *
 * @param dists 距离信息
 * @param k knn的值
 * @param frames 对应的索引
 */
void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k,
                                       std::vector<int> frames) {

  // 确保dist不为空
  if (!dists.size()) {
    return;
  }

  // 维护最多包含 k
  // 个元素的最大堆,由于d是当前帧和所有关键帧求得,所以一开始的关键帧应该距离更远,如果处于回环的时候,有可能就达不到K个的要求
  std::priority_queue<float> pq;

  for (auto d : dists) {
    //一直和堆顶比较,如果比堆顶小,就弹出堆顶,压入新的
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // 获取第k小的元素，它应该在堆的顶部
  float kth_element = pq.top();

  // 获取所有小于或等于第k小元素的元素进行压入
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }
}

void dlio::OdomNode::buildSubmap(State vehicle_state) {

  // 清除用于子地图的关键帧索引向量
  this->submap_kf_idx_curr.clear();

  // 计算当前姿态与关键帧集合中的姿态之间的距离
  std::unique_lock<decltype(this->keyframes_mutex)> lock(
      this->keyframes_mutex); //通过decltype关键字可以获得变量的类型,并加上互斥锁
  std::vector<float> ds; //用于存储当前帧与关键帧之间的距离
  std::vector<int> keyframe_nn; //用于存储当前帧与关键帧之间的索引
  for (int i = 0; i < this->num_processed_keyframes;
       i++) { //获取当前时刻所有的关键帧
    float d =
        sqrt(pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
             pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
             pow(vehicle_state.p[2] - this->keyframes[i].first.first[2],
                 2));         //计算当前帧与关键帧之间的距离
    ds.push_back(d);          //将距离存入ds
    keyframe_nn.push_back(i); //将索引存入keyframe_nn
  }
  lock.unlock();

  // 获取前K个最近邻关键帧姿态的索引
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // 获取凸包索引,其实就是提取一些不必要的关键帧
  this->computeConvexHull();

  // 获取凸包上每个关键帧之间的距离
  std::vector<float> convex_ds;
  for (const auto &c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]); //根据对应的索引将结果压入
  }

  // 获取凸包的前k个最近邻的索引
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // 获取凹包索引,其实就是提取一些不必要的关键帧
  this->computeConcaveHull();

  // 获取凸包上每个关键帧之间的距离
  std::vector<float> concave_ds;
  for (const auto &c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }

  // 获取凹包的前k个最近邻的索引
  this->pushSubmapIndices(concave_ds, this->submap_kcc_,
                          this->keyframe_concave);

  // 连接所有子地图的点云和法向量
  std::sort(this->submap_kf_idx_curr.begin(),
            this->submap_kf_idx_curr.end()); //对当前帧的索引进行排序
  auto last = std::unique(this->submap_kf_idx_curr.begin(),
                          this->submap_kf_idx_curr.end()); //去除重复的元素
  this->submap_kf_idx_curr.erase(
      last, this->submap_kf_idx_curr.end()); //删除重复的元素

  // 对当前和之前的子地图的索引列表进行排序
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // 检查子地图是否与上一次迭代时发生了变化
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev) {

    this->submap_hasChanged = true; //如果发生了变化,则将标志位置为true

    // 暂停以防止从主循环中窃取资源，如果主循环正在运行。
    this->pauseSubmapBuildIfNeeded();

    // 重新初始化子地图云和法线
    pcl::PointCloud<PointType>::Ptr submap_cloud_(
        boost::make_shared<pcl::PointCloud<PointType>>());
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_(
        std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) { //遍历当前帧的索引

      // 创建当前子地图云
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second; //将当前帧的点云压入
      lock.unlock();

      // 获取相应子地图云点的法向量
      submap_normals_->insert(std::end(*submap_normals_),
                              std::begin(*(this->keyframe_normals[k])),
                              std::end(*(this->keyframe_normals[k])));
    }

    this->submap_cloud = submap_cloud_; //将当前帧的点云赋值给子地图的点云
    this->submap_normals =
        submap_normals_; //将当前帧的法向量赋值给子地图的法向量

    // 如果主循环正在运行，请暂停以防止窃取资源
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(
        this->submap_cloud); //将子地图的点云赋值给gicp_temp的目标点云
    this->submap_kdtree =
        this->gicp_temp
            .target_kdtree_; //将gicp_temp的目标点云的kd树赋值给子地图的kd树

    this->submap_kf_idx_prev =
        this->submap_kf_idx_curr; //  将当前帧的索引赋值给上一帧的索引
  }
}

/**
 * @brief 通过这个函数完成子图的创建,其中包括了子图的点云和法线的创建
 *
 * @param vehicle_state 当前的车辆状态
 */
void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // 转换新的关键帧和相关的协方差列表
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size();
       i++) { //遍历未处理的关键帧,并完成局部点云向全局点云的转换
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe =
        this->keyframes[i].second; //获取关键帧的点云
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances =
        this->keyframe_normals[i]; //获取关键帧的协防差
    Eigen::Matrix4f T =
        this->keyframe_transformations[i]; //获取关键帧的变换矩阵
    lock.unlock();

    Eigen::Matrix4d Td =
        T.cast<double>(); //将float类型的变换矩阵转换为double类型的变换矩阵

    pcl::PointCloud<PointType>::Ptr transformed_keyframe(
        boost::make_shared<pcl::PointCloud<PointType>>());
    pcl::transformPointCloud(*raw_keyframe, *transformed_keyframe,
                             T); //将关键帧点云转换到世界坐标系下

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances(
        std::make_shared<nano_gicp::CovarianceList>(
            raw_covariances->size())); //创建一个新的协方差列表
    std::transform(raw_covariances->begin(), raw_covariances->end(),
                   transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) {
                     return Td * cov * Td.transpose();
                   }); //将关键帧的协方差转换到世界坐标系下

    ++this->num_processed_keyframes; //更新已经处理的关键帧的数量

    lock.lock();
    this->keyframes[i].second = transformed_keyframe; //更新关键帧的点云
    this->keyframe_normals[i] =
        transformed_covariances; //更新关键帧的协方差(法向量)

    this->publish_keyframe_thread =
        std::thread(&dlio::OdomNode::publishKeyframe, this, this->keyframes[i],
                    this->keyframe_timestamps[i]); //发布关键帧
    this->publish_keyframe_thread.detach();
  }

  lock.unlock();

  // 暂停以防止从主循环中窃取资源，如果主循环正在运行
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state); //创建子图
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(
      this->main_loop_running_mutex); //创建一个互斥锁
  // 在等待过程中，当前线程会自动释放锁lock，以允许其他线程对受保护的资源进行访问。等待结束后，当前线程会重新获得锁lock，并继续执行后面的代码。
  this->submap_build_cv.wait(lock, [this] {
    return !this->main_loop_running;
  }); //等待主循环结束,然后再继续执行
}

void dlio::OdomNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto &t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l =
        sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) +
             pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
      std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) /
      this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
        std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) /
        this->imu_rates.size();
  } else {
    avg_imu_rate = std::accumulate(this->imu_rates.end() - win_size,
                                   this->imu_rates.end(), 0.0) /
                   win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate = std::accumulate(this->lidar_rates.begin(),
                                     this->lidar_rates.end(), 0.0) /
                     this->lidar_rates.size();
  } else {
    avg_lidar_rate = std::accumulate(this->lidar_rates.end() - win_size,
                                     this->lidar_rates.end(), 0.0) /
                     win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat",
                            std::ios_base::in); // get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice >> num_threads >>
      itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
    cpu_percent = -1.0;
  } else {
    cpu_percent = (timeSample.tms_stime - this->lastSysCPU) +
                  (timeSample.tms_utime - this->lastUserCPU);
    cpu_percent /= (now - this->lastCPU);
    cpu_percent /= this->numProcessors;
    cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage = std::accumulate(this->cpu_percents.begin(),
                                         this->cpu_percents.end(), 0.0) /
                         this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout
      << std::endl
      << "+-------------------------------------------------------------------+"
      << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v"
            << this->version_ << "               |" << std::endl;
  std::cout
      << "+-------------------------------------------------------------------+"
      << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time));
  asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
            << "Elapsed Time: " +
                   to_string_with_precision(this->elapsed_time, 2) + " seconds "
            << "|" << std::endl;

  if (!this->cpu_type.empty()) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << this->cpu_type + " x " + std::to_string(this->numProcessors)
              << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Ouster @ " +
                     to_string_with_precision(avg_lidar_rate, 2) +
                     " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) +
                     " Hz"
              << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: Velodyne @ " +
                     to_string_with_precision(avg_lidar_rate, 2) +
                     " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) +
                     " Hz"
              << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
              << "Sensor Rates: LiDAR @ " +
                     to_string_with_precision(avg_lidar_rate, 2) +
                     " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) +
                     " Hz"
              << "|" << std::endl;
  }

  std::cout
      << "|===================================================================|"
      << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Position     {W}  [xyz] :: " +
                   to_string_with_precision(this->state.p[0], 4) + " " +
                   to_string_with_precision(this->state.p[1], 4) + " " +
                   to_string_with_precision(this->state.p[2], 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Orientation  {W} [wxyz] :: " +
                   to_string_with_precision(this->state.q.w(), 4) + " " +
                   to_string_with_precision(this->state.q.x(), 4) + " " +
                   to_string_with_precision(this->state.q.y(), 4) + " " +
                   to_string_with_precision(this->state.q.z(), 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Lin Velocity {B}  [xyz] :: " +
                   to_string_with_precision(this->state.v.lin.b[0], 4) + " " +
                   to_string_with_precision(this->state.v.lin.b[1], 4) + " " +
                   to_string_with_precision(this->state.v.lin.b[2], 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Ang Velocity {B}  [xyz] :: " +
                   to_string_with_precision(this->state.v.ang.b[0], 4) + " " +
                   to_string_with_precision(this->state.v.ang.b[1], 4) + " " +
                   to_string_with_precision(this->state.v.ang.b[2], 4)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Accel Bias        [xyz] :: " +
                   to_string_with_precision(this->state.b.accel[0], 8) + " " +
                   to_string_with_precision(this->state.b.accel[1], 8) + " " +
                   to_string_with_precision(this->state.b.accel[2], 8)
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Gyro Bias         [xyz] :: " +
                   to_string_with_precision(this->state.b.gyro[0], 8) + " " +
                   to_string_with_precision(this->state.b.gyro[1], 8) + " " +
                   to_string_with_precision(this->state.b.gyro[2], 8)
            << "|" << std::endl;

  std::cout
      << "|                                                                   |"
      << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Distance Traveled  :: " +
                   to_string_with_precision(length_traversed, 4) + " meters"
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Distance to Origin :: " +
                   to_string_with_precision(
                       sqrt(pow(this->state.p[0] - this->origin[0], 2) +
                            pow(this->state.p[1] - this->origin[1], 2) +
                            pow(this->state.p[2] - this->origin[2], 2)),
                       4) +
                   " meters"
            << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "Registration       :: keyframes: " +
                   std::to_string(this->keyframes.size()) + ", " +
                   "deskewed points: " + std::to_string(this->deskew_size)
            << "|" << std::endl;
  std::cout
      << "|                                                                   |"
      << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: " << std::setfill(' ') << std::setw(6)
            << this->comp_times.back() * 1000.
            << " ms    // Avg: " << std::setw(6) << avg_comp_time * 1000.
            << " / Max: " << std::setw(6)
            << *std::max_element(this->comp_times.begin(),
                                 this->comp_times.end()) *
                   1000.
            << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: " << std::setfill(' ') << std::setw(6)
            << (cpu_percent / 100.) * this->numProcessors
            << " cores // Avg: " << std::setw(6)
            << (avg_cpu_usage / 100.) * this->numProcessors
            << " / Max: " << std::setw(6)
            << (*std::max_element(this->cpu_percents.begin(),
                                  this->cpu_percents.end()) /
                100.) *
                   this->numProcessors
            << "     |" << std::endl;
  std::cout << "| CPU Load         :: " << std::setfill(' ') << std::setw(6)
            << cpu_percent << " %     // Avg: " << std::setw(6) << avg_cpu_usage
            << " / Max: " << std::setw(6)
            << *std::max_element(this->cpu_percents.begin(),
                                 this->cpu_percents.end())
            << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
            << "RAM Allocation   :: " +
                   to_string_with_precision(resident_set / 1000., 2) + " MB"
            << "|" << std::endl;

  std::cout
      << "+-------------------------------------------------------------------+"
      << std::endl;
}
