import serial
import threading
import time
import re
from .parser import WT901DataParser
from typing import Optional, List, Dict


class WT901Adapter:
    """维特智能蓝牙适配器控制类（通过串口AT指令）"""

    def __init__(self, port: str, baudrate: int = 115200, debug: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.debug = debug                # 是否打印发送的指令
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.parser = WT901DataParser()
        self.running = False
        self.recv_thread: Optional[threading.Thread] = None
        self._response_buffer = ""
        self._last_scan_text = ""
        self._last_read_text = ""
        self._last_binding_text = ""
        self._last_connect_text = ""
        self._rx_bytes = 0
        self._rx_preview = bytearray()

        # 轮询相关
        self.polling = False
        self.poll_thread: Optional[threading.Thread] = None
        self.write_lock = threading.Lock()  # 串口写锁，避免多线程冲突

        # 注册默认回调
        self.parser.register_callback('acc_gyro_angle', self._on_acc_gyro_angle)
        self.parser.register_callback('magnetic', self._on_magnetic)
        self.parser.register_callback('temperature', self._on_temperature)
        self.parser.register_callback('battery', self._on_battery)
        self.parser.register_callback('quaternion', self._on_quaternion)

    # ---------- 电量百分比转换函数 ----------
    def _voltage_to_percent(self, raw_mv: int) -> int:
        """根据原始电压mV值返回电量百分比（基于官方对应表）"""
        if raw_mv > 396:
            return 100
        elif 393 <= raw_mv <= 396:
            return 90
        elif 387 <= raw_mv <= 393:
            return 75
        elif 382 <= raw_mv <= 387:
            return 60
        elif 379 <= raw_mv <= 382:
            return 50
        elif 377 <= raw_mv <= 379:
            return 40
        elif 373 <= raw_mv <= 377:
            return 30
        elif 370 <= raw_mv <= 373:
            return 20
        elif 368 <= raw_mv <= 370:
            return 15
        elif 350 <= raw_mv <= 368:
            return 10
        elif 340 <= raw_mv <= 350:
            return 5
        else:
            return 0

    # ---------- 回调函数 ----------
    def _on_acc_gyro_angle(self, data):
        acc = data['acc']
        gyro = data['gyro']
        angle = data['angle']
        print(f"\rACC(g): X={acc[0]:7.3f} Y={acc[1]:7.3f} Z={acc[2]:7.3f} | "
              f"GYR(°/s): X={gyro[0]:8.2f} Y={gyro[1]:8.2f} Z={gyro[2]:8.2f} | "
              f"ANG(°): X={angle[0]:7.2f} Y={angle[1]:7.2f} Z={angle[2]:7.2f}", end='', flush=True)

    def _on_magnetic(self, data):
        mag = data[1]
        print(f"\n[{time.strftime('%H:%M:%S')}] 🧲 MAG(mG): X={mag[0]:6d} Y={mag[1]:6d} Z={mag[2]:6d}")

    def _on_temperature(self, data):
        temp = data[1]
        print(f"\n[{time.strftime('%H:%M:%S')}] 🌡️ TEMP: {temp:5.2f} °C")

    def _on_battery(self, data):
        raw_mv, volt = data[1]  # 解包电压V
        percent = self._voltage_to_percent(raw_mv)
        print(f"\n[{time.strftime('%H:%M:%S')}] 🔋 BATTERY: {percent}% ({volt:.3f} V)")

    def _on_quaternion(self, data):
        q = data[1]
        print(f"\n[{time.strftime('%H:%M:%S')}] 🌀 QUAT: Q0={q[0]:.4f} Q1={q[1]:.4f} Q2={q[2]:.4f} Q3={q[3]:.4f}")

    # ---------- 串口操作 ----------
    def open(self) -> bool:
        """打开串口"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.5,
                write_timeout=0.5
            )
            # 设置DTR为True（对应C#的DtrEnable = true）
            self.ser.dtr = True
            time.sleep(0.1)  # 等待稳定
            print(f"✅ 串口 {self.port} 已打开")
            return True
        except Exception as e:
            print(f"❌ 打开串口失败: {e}")
            return False

    def close(self):
        """关闭串口"""
        self.stop_polling()
        self.running = False
        if self.recv_thread and self.recv_thread.is_alive():
            self.recv_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")

    # ---------- 发送指令（可选打印）----------
    def send_at(self, command: str):
        """发送AT指令（自动添加换行符）"""
        if not self.ser or not self.ser.is_open:
            print("串口未打开")
            return
        cmd = command + "\r\n"
        with self.write_lock:
            self.ser.write(cmd.encode())
        if self.debug:
            print(f"📤 AT指令: {command}")

    def send_hex(self, hex_str: str):
        """发送十六进制指令（如设置回传速率）"""
        hex_str = hex_str.replace(" ", "")
        if len(hex_str) % 2 != 0:
            print("❌ 十六进制字符串长度必须为偶数")
            return
        data = bytes.fromhex(hex_str)
        with self.write_lock:
            self.ser.write(data)
        if self.debug:
            print(f"📤 HEX: {hex_str}")

    def _read_window(self, timeout: float = 0.5) -> bytes:
        raw = bytearray()
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                raw.extend(self.ser.read(self.ser.in_waiting))
            time.sleep(0.05)
        return bytes(raw)

    def _track_rx(self, data: bytes) -> None:
        if not data:
            return
        self._rx_bytes += len(data)
        if len(self._rx_preview) < 64:
            remaining = 64 - len(self._rx_preview)
            self._rx_preview.extend(data[:remaining])
        self.parser.feed_data(data)

    def read_config(self, timeout: float = 0.8) -> str:
        if not self.ser or not self.ser.is_open:
            return ""
        self.ser.reset_input_buffer()
        self.send_at("AT+READ")
        raw = self._read_window(timeout)
        self._last_read_text = raw.decode(errors="ignore").strip()
        return self._last_read_text

    def set_binding(self, enabled: bool, timeout: float = 0.8) -> str:
        if not self.ser or not self.ser.is_open:
            return ""
        self.ser.reset_input_buffer()
        self.send_at(f"AT+BINDING={1 if enabled else 0}")
        raw = self._read_window(timeout)
        self._last_binding_text = raw.decode(errors="ignore").strip()
        return self._last_binding_text

    # ---------- 设备扫描 ----------
    def scan_devices(self, timeout: float = 5.0) -> List[Dict]:
        devices = []
        self._response_buffer = ""
        self.ser.reset_input_buffer()
        self.send_at("AT+SCAN=0")
        time.sleep(0.2)
        self.send_at("AT+SCAN=1")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                data = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
                self._response_buffer += data
            time.sleep(0.1)

        self.send_at("AT+SCAN=0")
        # The adapter often delivers the final "Scan_stop" line slightly after
        # the scan timeout loop exits. Drain it here so the next connect step
        # does not mistake that stale text for a connect-time response.
        stop_tail = self._read_window(0.8)
        if stop_tail:
            self._response_buffer += stop_tail.decode(errors="ignore")
        self._last_scan_text = self._response_buffer
        devices = self._parse_device_list(self._response_buffer)
        return devices

    def _parse_device_list(self, text: str) -> List[Dict]:
        patterns = [
            r'WIT-LIST-#\s*(\d+):"([^"]+)"\s+(0x[\dA-Fa-f]{12})\s+(-?\d+)',
            r'WIT-LIST-#\s*(\d+):([^"\s]+)\s+(0x[\dA-Fa-f]{12})\s+(-?\d+)',
            r'WIT-LIST-#\s*(\d+):"([^"]*)"\s+([0-9A-Fa-f:]{17})\s+(-?\d+)',
            r'WIT-LIST-#\s*(\d+):([^"\s]*)\s+([0-9A-Fa-f:]{17})\s+(-?\d+)'
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, re.MULTILINE))
        devices = []
        for match in matches:
            index = int(match[0])
            name = match[1].strip()
            addr = match[2]
            rssi = int(match[3])
            devices.append({
                'index': index,
                'name': name,
                'address': addr,
                'rssi': rssi
            })
        # 去重
        unique = {}
        for d in devices:
            if d['address'] not in unique:
                unique[d['address']] = d
        return list(unique.values())

    def get_last_scan_text(self) -> str:
        return self._last_scan_text

    # ---------- 连接设备 ----------
    def connect_device(self, device_index: int):
        if not self.ser or not self.ser.is_open:
            print("串口未打开")
            return False
        self._last_connect_text = ""
        self._rx_bytes = 0
        self._rx_preview.clear()
        self.ser.reset_input_buffer()
        self.send_at(f"AT+CONNECT={device_index}")
        time.sleep(0.1)
        self.send_at("AT+SCAN=0")
        early_raw = self._read_window(0.8)
        if early_raw:
            self._last_connect_text = early_raw.decode(errors="ignore").strip()
            self._track_rx(early_raw)
        if "ERROR" in self._last_connect_text.upper():
            self.connected = False
            return False
        self.connected = True
        print(f"🔗 正在连接设备 {device_index}...")
        self.running = True
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()
        time.sleep(2)
        self.start_polling(interval_ms=16.7)  # 20Hz轮询
        return True

    def _receive_loop(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting)
                    if data:
                        self._rx_bytes += len(data)
                        if len(self._rx_preview) < 64:
                            remaining = 64 - len(self._rx_preview)
                            self._rx_preview.extend(data[:remaining])
                        self.parser.feed_data(data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"\n❌ 接收错误: {e}")
                break
        self.connected = False

    def get_rx_diagnostics(self) -> Dict[str, object]:
        return {
            "rx_bytes": self._rx_bytes,
            "rx_preview_hex": self._rx_preview.hex(" "),
            "parser_stats": self.parser.get_stats(),
            "connected": self.connected,
            "polling": self.polling,
            "last_read_text": self._last_read_text,
            "last_binding_text": self._last_binding_text,
            "last_connect_text": self._last_connect_text.strip(),
        }

    # ---------- 20Hz轮询 ----------
    def start_polling(self, interval_ms: float = 16.7):
        if self.poll_thread and self.poll_thread.is_alive():
            return
        self.polling = True
        self.poll_thread = threading.Thread(target=self._polling_loop, args=(interval_ms / 1000.0,), daemon=True)
        self.poll_thread.start()
        print(f"🔄 已启动20Hz轮询（磁场/电量/四元数）")

    def stop_polling(self):
        self.polling = False
        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=1)

    def _polling_loop(self, interval: float):
        commands = [
            ("FF AA 27 3A 00", "磁场"),
            ("FF AA 27 64 00", "电量"),
            ("FF AA 27 51 00", "四元数")
        ]
        idx = 0
        while self.polling and self.running:
            hex_cmd, name = commands[idx % len(commands)]
            # 调用send_hex，但不打印（debug=False时不打印）
            self.send_hex(hex_cmd)
            idx += 1
            time.sleep(interval)

    # ---------- 传感器指令封装 ----------
    def query_magnetic(self):
        self.send_hex("FF AA 27 3A 00")

    def query_temperature(self):
        self.send_hex("FF AA 27 40 00")

    def query_battery(self):
        self.send_hex("FF AA 27 64 00")

    def query_quaternion(self):
        self.send_hex("FF AA 27 51 00")

    def set_rate_1hz(self):
        self.send_hex("FF AA 69 88 B5")
        time.sleep(0.05)
        self.send_hex("FF AA 03 03 00")
        time.sleep(0.05)
        self.send_hex("FF AA 00 00 00")

    def set_rate_10hz(self):
        self.send_hex("FF AA 69 88 B5")
        time.sleep(0.05)
        self.send_hex("FF AA 03 06 00")
        time.sleep(0.05)
        self.send_hex("FF AA 00 00 00")
