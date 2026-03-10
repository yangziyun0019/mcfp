import struct
from typing import Optional, List, Dict


class WT901DataParser:
    """WT9011DCL-BT50 传感器数据解析器（基于官方协议）"""

    # 寄存器地址常量
    REG_MAGNETIC = 0x3A      # 磁场
    REG_TEMP = 0x40          # 温度
    REG_QUATERNION = 0x51    # 四元数
    REG_BATTERY = 0x64       # 电量

    def __init__(self):
        self.buffer = bytearray()
        self.callbacks = {}
        self._last_acc = (0.0, 0.0, 0.0)
        self._last_gyro = (0.0, 0.0, 0.0)
        self._stats = {
            "feed_calls": 0,
            "raw_bytes": 0,
            "legacy_51": 0,
            "legacy_52": 0,
            "legacy_53": 0,
            "legacy_54": 0,
            "packet_61": 0,
            "packet_71": 0,
            "checksum_errors": 0,
            "dropped_prefix_bytes": 0,
            "unknown_flags": 0,
        }

    def register_callback(self, data_type: str, callback):
        """注册数据回调函数"""
        self.callbacks[data_type] = callback

    def _parse_short(self, low: int, high: int) -> int:
        """将低高字节组合成有符号16位整数"""
        val = (high << 8) | low
        return val if val < 32768 else val - 65536

    def feed_data(self, data: bytes):
        """向解析器喂数据"""
        self._stats["feed_calls"] += 1
        self._stats["raw_bytes"] += len(data)
        self.buffer.extend(data)
        self._parse_buffer()

    def _parse_buffer(self):
        """解析缓冲区中的数据包"""
        while len(self.buffer) >= 2:
            if self.buffer[0] != 0x55:
                self._stats["dropped_prefix_bytes"] += 1
                self.buffer.pop(0)
                continue

            flag = self.buffer[1]

            if flag in (0x51, 0x52, 0x53, 0x54):  # 原生 11 字节数据包
                if len(self.buffer) < 11:
                    break
                packet = self.buffer[:11]
                checksum = sum(packet[:10]) & 0xFF
                if checksum != packet[10]:
                    self._stats["checksum_errors"] += 1
                    self.buffer.pop(0)
                    continue
                self.buffer = self.buffer[11:]
                self._parse_packet_legacy(packet)

            elif flag == 0x61:  # 加速度/角速度/角度包 (20字节)
                if len(self.buffer) < 20:
                    break
                packet = self.buffer[:20]
                self.buffer = self.buffer[20:]
                self._stats["packet_61"] += 1
                self._parse_packet_61(packet[2:])

            elif flag == 0x71:  # 寄存器数据包 (20字节)
                if len(self.buffer) < 20:
                    break
                packet = self.buffer[:20]
                self.buffer = self.buffer[20:]
                self._stats["packet_71"] += 1
                self._parse_packet_71(packet[2:])

            else:
                self._stats["unknown_flags"] += 1
                self.buffer.pop(0)

    def _parse_packet_legacy(self, packet: bytes):
        """解析维特原生 11 字节数据包"""
        if len(packet) != 11:
            return

        flag = packet[1]
        data = packet[2:10]

        if flag == 0x51:  # 加速度
            self._stats["legacy_51"] += 1
            ax = self._parse_short(data[0], data[1]) / 32768.0 * 16.0
            ay = self._parse_short(data[2], data[3]) / 32768.0 * 16.0
            az = self._parse_short(data[4], data[5]) / 32768.0 * 16.0
            self._last_acc = (ax, ay, az)

        elif flag == 0x52:  # 角速度
            self._stats["legacy_52"] += 1
            wx = self._parse_short(data[0], data[1]) / 32768.0 * 2000.0
            wy = self._parse_short(data[2], data[3]) / 32768.0 * 2000.0
            wz = self._parse_short(data[4], data[5]) / 32768.0 * 2000.0
            self._last_gyro = (wx, wy, wz)

        elif flag == 0x53:  # 欧拉角
            self._stats["legacy_53"] += 1
            roll = self._parse_short(data[0], data[1]) / 32768.0 * 180.0
            pitch = self._parse_short(data[2], data[3]) / 32768.0 * 180.0
            yaw = self._parse_short(data[4], data[5]) / 32768.0 * 180.0
            result = {
                'type': 'acc_gyro_angle',
                'acc': self._last_acc,
                'gyro': self._last_gyro,
                'angle': (roll, pitch, yaw)
            }
            if 'acc_gyro_angle' in self.callbacks:
                self.callbacks['acc_gyro_angle'](result)

        elif flag == 0x54:  # 磁场
            self._stats["legacy_54"] += 1
            hx = self._parse_short(data[0], data[1])
            hy = self._parse_short(data[2], data[3])
            hz = self._parse_short(data[4], data[5])
            result = ('magnetic', (hx, hy, hz))
            if result[0] in self.callbacks:
                self.callbacks[result[0]](result)

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def _parse_packet_61(self, data: bytes):
        """解析0x61包 (加速度/角速度/角度)"""
        if len(data) < 18:
            return
        # 格式: <9个short (ax,ay,az, wx,wy,wz, roll,pitch,yaw)
        values = struct.unpack('<hhhhhhhhh', data[:18])
        ax, ay, az, wx, wy, wz, roll, pitch, yaw = values

        acc_scale = 16.0 / 32768.0
        gyro_scale = 2000.0 / 32768.0
        angle_scale = 180.0 / 32768.0

        result = {
            'type': 'acc_gyro_angle',
            'acc': (ax * acc_scale, ay * acc_scale, az * acc_scale),
            'gyro': (wx * gyro_scale, wy * gyro_scale, wz * gyro_scale),
            'angle': (roll * angle_scale, pitch * angle_scale, yaw * angle_scale)
        }
        if 'acc_gyro_angle' in self.callbacks:
            self.callbacks['acc_gyro_angle'](result)

    def _parse_packet_71(self, data: bytes):
        """解析0x71包 (寄存器读取返回)"""
        if len(data) < 18:
            return
        reg_addr = data[0] | (data[1] << 8)  # 起始寄存器地址
        reg_data = data[2:18]

        if reg_addr == self.REG_MAGNETIC:  # 磁场
            hx = self._parse_short(reg_data[0], reg_data[1])
            hy = self._parse_short(reg_data[2], reg_data[3])
            hz = self._parse_short(reg_data[4], reg_data[5])
            result = ('magnetic', (hx, hy, hz))
        elif reg_addr == self.REG_TEMP:  # 温度
            temp_raw = self._parse_short(reg_data[0], reg_data[1])
            result = ('temperature', temp_raw / 100.0)
        elif reg_addr == self.REG_QUATERNION:  # 四元数
            q0 = self._parse_short(reg_data[0], reg_data[1]) / 32768.0
            q1 = self._parse_short(reg_data[2], reg_data[3]) / 32768.0
            q2 = self._parse_short(reg_data[4], reg_data[5]) / 32768.0
            q3 = self._parse_short(reg_data[6], reg_data[7]) / 32768.0
            result = ('quaternion', (q0, q1, q2, q3))
        elif reg_addr == self.REG_BATTERY:  # 电量
            voltage_v = reg_data[0] | (reg_data[1] << 8)  # 原始mV值
            result = ('battery', (voltage_v, voltage_v / 100.0))  # 返回原始电压V
        else:
            result = ('unknown', reg_data)

        if result[0] in self.callbacks:
            self.callbacks[result[0]](result)
