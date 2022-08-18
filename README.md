# DeepStream Pose Estimation
## 1. 代码内容
* utils/utils.py: 功能函数，包含后处理相关函数
* utils/display.py: 功能函数, 包含把displayMeta加入到batchMeta中的函数
* pose.py: 只包含姿态估计推理
* pose_test01.py: 只包含姿态估计推理
* pose_test02.py: 包含姿态估计+跟踪+可视化, NMS放在后处理部分
* pose_test03.py: 包含姿态估计+跟踪+可视化, NMS放在模型中

## 2. 对比
&emsp;&emsp;如果要对比NMS在模型中与在模型外的DeepStream运行速度，只需分别运行pose_test02.py与pose_test03.py即可，二者的区别仅在于NMS处理部分, 运行前需要设置好视频路径 INPUT_STREAM , 如果要在 jeson nx 本地可视化结果, 需在运行上述python脚本时设置环境变量export DISPLAY=:0 指定显示设备为本地设备，将nx接个显示器即可与代码运行同步显示，或者设置代码中的DEBUG为True, 保持推理结果视频到本地
<details>
<summary>展开</summary>

step 1
```python
...
    # 在代码中指定推理视频路径
    INPUT_STREAM = ["file:///media/nvidia/SD/project/test/merge.mp4",]
...
```

step 2
```bash
# 设置本地显示
export DISPLAY=:0
python3 pose_test02.py
python3 pose_test03.py
```
</details>

## 3. 后处理函数(utils/utils.py)
preprocessNoNMS: NMS融入模型后的后处理函数  
preprocess: NMS未融入模型的后处理函数