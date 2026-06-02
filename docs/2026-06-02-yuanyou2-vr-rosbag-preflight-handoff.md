# Yuanyou2 VR 遥操与 Rosbag 采集前置工作交接

最后更新：2026-06-02  
远端主机：`yuanyou@192.168.31.230`  
工作空间：`/home/yuanyou/yuanyou2_ws`

## 1. 当前结论

正式遥操或采集前，必须按下面顺序启动：

```text
CAN -> Piper 双臂底层 -> 三路相机 -> 相机 TF -> VR 输入服务 -> VR 控制副本 -> rosbag 录制
```

网页 `http://192.168.31.230:5000/` 只负责 rosbag 的开始/停止记录，不会自动启动相机，也不会自动启动机械臂。

## 2. 已保护的原版文件

以下原版文件不要改，当前优化都放在副本里：

```text
/home/yuanyou/yuanyou2_ws/src/yuanyou2/scripts/right_vr_pose_controller.py
/home/yuanyou/yuanyou2_ws/src/yuanyou2/scripts/left_vr_pose_controller.py
/home/yuanyou/yuanyou2_ws/src/yuanyou2/launch/vr_double.launch
```

已记录的原版 hash：

```text
490bef85bb0a4fba234fb4d0b3ebb926b3161cf80aa9e893254e48321c3722cc  right_vr_pose_controller.py
9230aaa1e8eb3e361eff301ffffb99059abbf41f70e390b0789d112908c0ad63  left_vr_pose_controller.py
99f88e2dc3f51a026adceb840527f255f9c891026c95e29a7abcf207753ecf5b  vr_double.launch
```

优化副本路径：

```text
/home/yuanyou/yuanyou2_ws/src/yuanyou2/scripts/right_vr_pose_controller_debounce.py
/home/yuanyou/yuanyou2_ws/src/yuanyou2/scripts/left_vr_pose_controller_debounce.py
/home/yuanyou/yuanyou2_ws/src/yuanyou2/launch/vr_double_debounce.launch
```

优化副本备份目录：

```text
/home/yuanyou/vr_double_backup_20260601_debounce_gain/
```

当前优化副本只做滤波、静止去抖和输出倍率参数。`vr_double_debounce.launch` 当前默认：

```text
output_gain=1.05
input_pos_alpha=0.65
input_rot_alpha=0.65
target_pos_hold_deadband=0.0008
```

## 3. 终端 1：启动 CAN

如果 `can0/can1` 是 `DOWN`，左右 Piper 节点会直接退出，`joint_remapper_node` 会一直打印：

```text
left_ready=False, right_ready=False
```

启动 CAN：

```bash
sudo bash /home/yuanyou/yuanyou2_ws/src/piper_sdk/piper_sdk/can_muti_activate.sh
```

检查：

```bash
ip -details link show can0
ip -details link show can1
```

应看到：

```text
state UP
bitrate 1000000
```

当前硬件映射：

```text
can0 bus-info: 1-2.1.1:1.0
can1 bus-info: 1-2.1.3:1.0
left  -> can0
right -> can1
```

## 4. 终端 2：启动 Piper 双臂底层

```bash
source /opt/ros/noetic/setup.bash
source /home/yuanyou/yuanyou2_ws/devel/setup.bash
roslaunch piper start_double_piper.launch
```

检查：

```bash
rostopic hz /left/joint_states_single
rostopic hz /right/joint_states_single
```

这两个 topic 有频率后，再启动 VR 控制。否则先处理 CAN 或 Piper 节点日志。

## 5. 终端 3：启动左右腕 D435

当前两个 D435 正常枚举时，启动左右腕相机：

```bash
source /opt/ros/noetic/setup.bash
source /home/yuanyou/yuanyou2_ws/devel/setup.bash
roslaunch yuanyou2 three_cameras.launch start_left_wrist:=true start_right_wrist:=true start_head:=false
```

如果怀疑 D435 没接上：

```bash
rs-enumerate-devices -s
```

应看到两个序列号：

```text
342222072464
317422072133
```

## 6. 终端 4：启动头部 Orbbec 相机

头部相机当前按 Orbbec 启动，不用 `three_cameras.launch` 里的 `/dev/video12`。

```bash
source /opt/ros/noetic/setup.bash
source /home/yuanyou/yuanyou2_ws/devel/setup.bash
roslaunch orbbec_camera ob_camera.launch camera_name:=head_camera enable_point_cloud:=false enable_depth:=false publish_tf:=false
```

## 7. 终端 5：启动相机 TF

```bash
source /opt/ros/noetic/setup.bash
source /home/yuanyou/yuanyou2_ws/devel/setup.bash
roslaunch yuanyou2 camera_tf.launch
```

注意：腕部 TF 已按历史标定结果保存，不要重新标定或覆盖。头部相机挂在 `lift_link -> head_camera`。

## 8. 检查相机 topic

```bash
rostopic list | grep -E "left_wrist_d435|right_wrist_d435|head_camera|image_raw|camera_info"
```

当前 rosbag 记录用的真实图像 topic 是：

```bash
rostopic hz /left_wrist_d435/color/image_raw2
rostopic hz /right_wrist_d435/color/image_raw2
rostopic hz /head_camera/color/image_raw
```

这三个都有频率后，再开始 rosbag 录制。

## 9. 终端 6：启动 VR 输入服务

```bash
source /opt/ros/noetic/setup.bash
source /home/yuanyou/yuanyou2_ws/devel/setup.bash
cd /home/yuanyou/yuanyou2_ws/src/yuanyou2/scripts
python test_server.py
```

## 10. 终端 7：启动 VR 控制副本

建议先关闭吸盘，避免吸盘串口 `Input/output error` 干扰遥操测试：

```bash
source /opt/ros/noetic/setup.bash
source /home/yuanyou/yuanyou2_ws/devel/setup.bash
roslaunch yuanyou2 vr_double_debounce.launch enable_suction:=false
```

如果需要临时调整跟手程度：

```bash
roslaunch yuanyou2 vr_double_debounce.launch enable_suction:=false output_gain:=0.90
```

常用范围：

```text
output_gain=0.90  更稳，动作小
output_gain=1.05  当前默认
output_gain=1.20  更跟手，但更容易接近奇异点或限位
```

## 11. VR 操作顺序

控制器逻辑要求：

```text
idle/prep 状态下 reset_zero
reset_zero success 后再进入 teleop
```

如果在 `teleop` 状态按 reset，会被拒绝：

```text
ignore reset_zero because mode=teleop, only idle/prep allowed
```

如果看到：

```text
teleop skip: zero_robot_pos/quat not set, reset_zero first
```

说明还没有成功记录零点，需要回到 `idle` 后重新 reset_zero。

## 12. Rosbag 网页录制

网页地址：

```text
http://192.168.31.230:5000/
```

如果浏览器显示 `502 Bad Gateway`，通常是电脑代理把局域网地址也走代理了。关闭代理或设置绕过局域网。也可以走 SSH 转发：

```bash
ssh -L 5000:127.0.0.1:5000 yuanyou@192.168.31.230
```

然后访问：

```text
http://127.0.0.1:5000/
```

录制脚本已修复，会在子进程里 source ROS 环境后再启动 rosbag。备份文件：

```text
/home/yuanyou/robot_bag_web/app.py.backup_20260601_camera_topics
/home/yuanyou/robot_bag_web/app.py.backup_20260601_rosbag_abs
/home/yuanyou/robot_bag_web/app.py.backup_20260601_rosbag_env
```

当前录制 topic 列表：

```text
/joint_states
/left/joint_cmd
/right/joint_cmd
/left_wrist_d435/color/image_raw2
/left_wrist_d435/color/camera_info
/right_wrist_d435/color/image_raw2
/right_wrist_d435/color/camera_info
/head_camera/color/image_raw
/head_camera/color/camera_info
/tf
/tf_static
```

录制后的检查：

```bash
source /opt/ros/noetic/setup.bash
rosbag info /home/yuanyou/robot_dataset_rosbag/<episode_dir>/<episode_name>.bag
```

确认至少有：

```text
/left_wrist_d435/color/image_raw2
/right_wrist_d435/color/image_raw2
/head_camera/color/image_raw
/joint_states
```

## 13. 常见故障

### Piper 节点退出

现象：

```text
piper_ctrl_left_node process has died
piper_ctrl_right_node process has died
joint_remapper_node waiting for messages
```

优先检查：

```bash
ip -details link show can0
ip -details link show can1
```

若 CAN 是 `DOWN / STOPPED`，重新执行 CAN 启动脚本。

### 相机没图像

检查硬件：

```bash
rs-enumerate-devices -s
```

检查 topic：

```bash
rostopic list | grep -E "image_raw|camera_info|d435|head_camera"
```

注意：网页录制不会启动相机，相机必须先由 ROS launch 启动。

### 吸盘串口报错

现象：

```text
[suction_serial_node] status failed: (5, 'Input/output error')
```

先排除干扰：

```bash
roslaunch yuanyou2 vr_double_debounce.launch enable_suction:=false
```

之后再单独检查吸盘串口：

```bash
ls -l /dev/serial/by-id
ls -l /dev/ttyUSB*
```

当前吸盘默认端口：

```text
/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

### 到奇异点或限位附近乱动

这是机械臂 IK/机构边界问题，不是单纯滤波问题。先降低输出倍率：

```bash
roslaunch yuanyou2 vr_double_debounce.launch enable_suction:=false output_gain:=0.90
```

同时避免把 VR 目标推到基座前方过深、手腕姿态过大或机械臂明显接近伸直/折叠的位置。

## 14. 快速启动清单

每次正式采集前确认：

```text
[ ] can0/can1 UP, bitrate 1000000
[ ] /left/joint_states_single 有频率
[ ] /right/joint_states_single 有频率
[ ] /left_wrist_d435/color/image_raw2 有频率
[ ] /right_wrist_d435/color/image_raw2 有频率
[ ] /head_camera/color/image_raw 有频率
[ ] test_server.py 已启动
[ ] vr_double_debounce.launch 已启动
[ ] reset_zero success 后再进入 teleop
[ ] 5000 网页开始 rosbag 录制
```
