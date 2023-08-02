# RNN-SLAM
#Probar DSO en RNNSLAM
./bin/dso_dataset mode=2 preset=0 files=~/Desktop/Sequences/rnnslam_real_data_2_enhanced/images calib=~/Desktop/Sequences/rnnslam_real_data_2_enhanced/camera.txt rnn=~/Desktop/RNNSLAM/src/RNN rnnmodel=~/Desktop/RNNSLAM/src/RNN/models/model-145000 numRNNBootstrap=9 lostTolerance=5 output_prefix=~/Desktop/Sequences/rnnslam_real_data_2_enhanced/out/ quiet=1 sampleoutput=1


#Convertir a formato TUM
python cvt_colon_to_tumrgbd.py --depth_dir ~/Desktop/Sequences/rnnslam_real_data_2_enhanced/out/depth --image_dir ~/Desktop/Sequences/rnnslam_real_data_2_enhanced/images --cameras_file_path ~/Desktop/Sequences/rnnslam_real_data_2_enhanced/out/kf_pose_result.txt --output_dir ~/Desktop/Sequences/rnnslam_real_data_2_enhanced/tum --intrinsic ~/Desktop/Sequences/rnnslam_real_data_2_enhanced/camera.txt --repeat 1 --high_intensity_threshold 250 --low_intensity_threshold 70 --rescale_w 320 --rescale_h 256


#Crear malla
./build/applications/surfel_meshing/SurfelMeshing ~/Desktop/Sequences/rnnslam_real_data_2_enhanced/tum trajectory.txt --follow_input_camera true --depth_valid_region_radius 160 --export_mesh mesh_031.obj --outlier_filtering_frame_count 2 --outlier_filtering_required_inliers 1 --observation_angle_threshold_deg 90 --sensor_noise_factor 0.3 --hide_camera_frustum --max_depth 2.5 --bilateral_filter_radius_factor 5


#Versiones 
Ubuntu 18.04
CUDA 10.1
Pangolin 0.6
Python 3.7.5
Tensorflow 2.11


