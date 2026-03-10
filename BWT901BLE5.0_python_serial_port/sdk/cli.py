import serial.tools.list_ports
from .adapter import WT901Adapter


def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def main():
    print("=" * 60)
    print("   WT9011DCL-BT50 蓝牙适配器控制程序")
    print("   - 实时显示加速度/角速度/角度")
    print("   - 自动20Hz轮询磁场、电量（百分比）、四元数")
    print("=" * 60)

    # 1. 选择串口
    ports = list_serial_ports()
    if not ports:
        print("❌ 未找到串口")
        return

    print("\n可用串口：")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")

    choice = input("请选择串口编号: ").strip()
    if not choice.isdigit() or int(choice) >= len(ports):
        print("❌ 无效选择")
        return
    port = ports[int(choice)]

    # 2. 打开串口（debug=False 不打印发送的指令）
    adapter = WT901Adapter(port, debug=True)
    if not adapter.open():
        return

    # 3. 扫描设备
    input("\n按 Enter 开始扫描蓝牙设备...")
    devices = adapter.scan_devices(timeout=12)

    if not devices:
        print("⚠️ 未扫描到任何设备")
        raw = adapter.get_last_scan_text().strip()
        if raw:
            print("\n适配器原始扫描响应：")
            print(raw)
        else:
            print("\n适配器没有返回任何扫描数据。")
            print("请优先检查：")
            print("  1. 选择的是否是蓝牙适配器串口，而不是 Arduino 串口")
            print("  2. 传感器是否已开机且未被手机或其他设备占用")
            print("  3. 适配器和传感器是否靠近")
        adapter.close()
        return

    # 过滤 WT 开头的设备
    wt_devices = [d for d in devices if d['name'].upper().startswith('WT')]
    if wt_devices:
        print("\n📋 WT 设备列表：")
        for i, d in enumerate(wt_devices):
            print(f"  [{i}] {d['name']} - {d['address']} (RSSI: {d['rssi']} dBm)")
    else:
        print("⚠️ 未找到 WT 开头的设备，显示所有：")
        for i, d in enumerate(devices):
            print(f"  [{i}] {d['name']} - {d['address']} (RSSI: {d['rssi']} dBm)")
        wt_devices = devices

    # 4. 选择设备连接
    choice = input("\n请输入要连接的设备编号: ").strip()
    if not choice.isdigit() or int(choice) >= len(wt_devices):
        print("❌ 无效选择")
        adapter.close()
        return
    selected = wt_devices[int(choice)]

    print(f"正在连接 {selected['name']} ...")
    if not adapter.connect_device(selected['index']):
        print("❌ 连接失败")
        adapter.close()
        return

    # 5. 交互命令
    print("\n🚀 连接成功，开始接收数据（按 Ctrl+C 退出）")
    print("📋 可用指令：")
    print("  1 - 读取温度（单次）")
    print("  5 - 设置回传速率 1Hz")
    print("  6 - 设置回传速率 10Hz")
    print("  h - 发送自定义十六进制指令")
    print("  q - 退出")

    try:
        while True:
            cmd = input("\n指令 > ").strip().lower()
            if cmd == '1':
                adapter.query_temperature()
            elif cmd == '5':
                adapter.set_rate_1hz()
            elif cmd == '6':
                adapter.set_rate_10hz()
            elif cmd == 'h':
                hex_str = input("请输入十六进制字符串（如 FFAA6988B5）: ").strip()
                adapter.send_hex(hex_str)
            elif cmd == 'q':
                break
            else:
                print("❓ 未知指令")
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    finally:
        adapter.close()
