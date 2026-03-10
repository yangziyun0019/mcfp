# 遥操作采集工程说明

这个工程用于通过遥操作方式采集机械臂控制数据，当前主入口脚本是：

- `host/arx_x5_teleop.py`

它的基本思路是：下位机采集电位器和按键信号，IMU 提供姿态信息，主机侧将这些输入直接映射成机械臂控制指令并下发，不经过高层规划或学习模型。

## 当前工程包含的内容

### 1. 机械结构

- `structure/`
  - 包含主臂的机械结构建模文件
  - 包含 SolidWorks 零件、装配体、STEP 文件
  - 包含用于 3D 打印的 STL 文件

这部分主要用于：

- 结构设计留档
- 打印件复现
- 后续机械改型对比

### 2. Arduino 采集部分

- `src/main.cpp`
  - Arduino 端主程序
- `platformio.ini`
  - PlatformIO 工程配置

当前下位机方案是：

- Arduino 板子
- 电位器采集关节输入
- 按键采集夹爪开合输入

Arduino 端会周期性通过串口输出控制帧，供主机侧读取。

### 3. IMU 数据采集

- `BWT901BLE5.0_python_serial_port/`
  - IMU 相关的串口采集和适配代码
- `host/imu_vendor_adapter.py`
  - Host 侧对 IMU 厂商接口的适配
- `host/imu_sources.py`
  - IMU 数据源封装

当前 IMU 主要提供：

- `roll`
- `pitch`
- `yaw`

### 4. Host 端遥操作逻辑

- `host/arx_x5_teleop.py`
  - 当前遥操作主脚本
- `host/teleop_input.py`
  - 负责将 Arduino 和 IMU 输入合并成统一控制帧
- `host/arx_x5_safe_driver.py`
  - 负责将控制帧映射为机械臂关节和夹爪命令

当前控制方式是**直接映射**：

- 电位器角度直接映射到部分关节
- IMU 姿态角直接映射到部分关节
- 按键直接映射为夹爪开合

也就是说，这个工程当前不是做自主规划，而是做“人操作输入 -> 机械臂关节命令”的直接控制。

## 主要数据流

1. Arduino 读取电位器和按键
2. Arduino 通过串口发送下位机控制帧
3. IMU 通过串口输出姿态角
4. Host 读取两路输入并合并
5. `host/arx_x5_teleop.py` 根据映射关系生成目标关节指令
6. Host 通过 ARX SDK 直接向机械臂发送控制命令

## 常用运行方式

### 1. 烧录 Arduino 程序

```bash
pio run -t upload
```

串口监视：

```bash
pio device monitor
```

### 2. 配置 Host Python 环境

推荐使用独立 venv：

```bash
/usr/bin/python3 -m venv .venv-arx
source .venv-arx/bin/activate
python -m pip install -r host/requirements.txt
```

### 3. 启动遥操作

主入口：

```bash
python -m host.arx_x5_teleop
```

如果需要显式指定用于兼容 ARX SDK 的 Python：

```bash
ARX_TELEOP_PYTHON=$(pwd)/.venv-arx/bin/python python -m host.arx_x5_teleop
```

### 4. 仅检查输入链路

如果只想检查 Arduino + IMU 输入是否正常合并：

```bash
python -m host.run_teleop
```

## 目录说明

- `host/`
  - 主机侧遥操作与驱动逻辑
- `src/`
  - Arduino 固件源码
- `structure/`
  - 机械结构建模与 3D 打印文件
- `BWT901BLE5.0_python_serial_port/`
  - IMU 串口读取相关代码
- `ARX_X5-main/`
  - ARX 相关 SDK / 厂商工程

## 当前定位

这个工程当前主要用于：

- 遥操作数据采集
- 控制链路联调
- 机械结构与电子硬件联合验证

后续如果要扩展，也比较适合继续加入：

- 数据记录
- 控制日志导出
- 安全限幅
- 遥操作数据回放
