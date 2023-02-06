# cable_observer
<!-- Required -->
<!-- Package description -->
A node for DLO tracking.

## Installation
<!-- Required -->
<!-- Things to consider:
    - How to build package? 
    - Are there any other 3rd party dependencies required? -->

```bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-up-to cable_observer
```

## Usage
<!-- Required -->
<!-- Things to consider:
    - Launching package. 
    - Exposed API (example service/action call. -->

```bash
ros2 launch cable_observer cable_observer.launch.py with_rviz:=True
```

## API
<!-- Required -->
<!-- Things to consider:
    - How do you use the package / API? -->


### Input

| Name                      | Type                         | Description                         |
| ------------------------- | ---------------------------- | ----------------------------------- |
| `/rgb/camera_info`        | sensor_msgs::msg::CameraInfo | RGB camera info.                    |
| `/rgb/image_raw`          | sensor_msgs::msg::Image      | RGB image.                          |
| `/depth_to_rgb/image_raw` | sensor_msgs::msg::Image      | Depth image (aligned to RGB image). |

### Output

| Name                     | Type                             | Description               |
| ------------------------ | -------------------------------- | ------------------------- |
| `/cable_observer/marker` | visualization_msgs::msg::Marker  | DLO visualization.        |
| `/cable_observer/coords` | std_msgs::msg::Float64MultiArray | DLO coordinates (x, y, z) |


### Parameters

| Name               | Type      | Description                                                  |
| ------------------ | --------- | ------------------------------------------------------------ |
| `debug`            | bool      | Print durations.                                             |
| `depth_ranges`     | list[int] | Depth region of interest.                                    |
| `depth_scale`      | float     | Depth scalling factor (expecting meters).                    |
| `hsv_ranges`       | list[int] | HSV color ranges [h_min, s_min, v_min, h_max, s_max, v_max]  |
| `min_length`       | int       | Minimum lenght (euclidean pxs) for partial paths.            |
| `num_of_knots`     | int       | Number of knots for output spline.                           |
| `num_of_pts`       | int       | Number of sampled points for output spline.                  |
| `vector_dir_len`   | int       | Number of points which describe path direction on path ends. |
| `z_vertical_shift` | int       | Vertical shift (pxs) between depth and color input           |


## References / External links
<!-- Optional -->
