#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WT9011DCL-BT50 蓝牙适配器控制与数据解析程序
- 实时显示加速度/角速度/角度（同一行刷新）
- 自动20Hz轮询磁场、电量、四元数并换行打印
- 电量显示改为百分比（根据官方电压-百分比对应表）
"""

from sdk import cli_main


if __name__ == "__main__":
    cli_main()
