# OpenICC


```bash
pixi run install-theia
pixi run build
pixi run python ./OpenImuCameraCalibrator/python/static_multipose_imu_calibration.py --path_static_calib_dataset=GoPro9/imu_bias --initial_static_duration_s=10 --path_to_build=build/OpenImuCameraCalibrator/applications/
```
