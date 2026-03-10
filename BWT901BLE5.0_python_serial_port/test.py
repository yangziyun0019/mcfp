import asyncio

import device_model

# Fixed MAC from vendor.
TARGET_MAC = "DE:21:D0:72:09:E2"
# Retry interval when connect fails/disconnects.
RETRY_SECONDS = 2.0


def updateData(DeviceModel):
    print(DeviceModel.deviceData)


async def run_forever() -> None:
    attempt = 0
    while True:
        attempt += 1
        print(f"[attempt {attempt}] connecting to {TARGET_MAC} ...")
        device = device_model.DeviceModel("MyBluetoothDevice", TARGET_MAC, updateData)
        try:
            await device.openDevice()
            print("Disconnected. Reconnecting soon...")
        except Exception as ex:
            print(f"Connect failed: {ex}")
        await asyncio.sleep(RETRY_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        print("\nStopped by user")
