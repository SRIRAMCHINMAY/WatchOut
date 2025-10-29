import serial
import time
import re
import pandas as pd
import numpy as np


def process_packets(log_text: str, total_duration: float = 60.0):
    """
    Parse packet log text, extract key fields, and assign synthetic timing information.
    """
    packets_raw = re.split(r'---- PACKET ----', log_text)
    packets = []

    for pkt in packets_raw:
        if not pkt.strip():
            continue

        # Extract fields
        rssi = re.search(r'RSSI:\s*(-?\d+)', pkt)
        channel = re.search(r'Channel:\s*(\d+)', pkt)
        length = re.search(r'Len:\s*(\d+)', pkt)
        da = re.search(r'DA:\s*([\dA-F:]+)', pkt, re.I)
        sa = re.search(r'SA:\s*([\dA-F:]+)', pkt, re.I)
        bssid = re.search(r'BSSID:\s*([\dA-F:]+)', pkt, re.I)

        packets.append({
            "radiotap.dbm_antsignal": int(rssi.group(1)) if rssi else None,
            "radiotap.channel.freq": int(channel.group(1)) if channel else None,
            "frame.len": int(length.group(1)) if length else None,
            "frame.cap_len": int(length.group(1)) if length else None,
            "wlan.da": da.group(1) if da else None,
            "wlan.sa": sa.group(1) if sa else None,
            "wlan.bssid": bssid.group(1) if bssid else None,
        })

    df = pd.DataFrame(packets)
    n = len(df)

    if n > 1:
        df["frame.time_relative"] = np.linspace(0, total_duration, n)
        df["frame.time_delta"] = df["frame.time_relative"].diff().fillna(0)
    else:
        df["frame.time_relative"] = 0
        df["frame.time_delta"] = 0

    return df


def read_serial_packets(port="/dev/ttyUSB0", baudrate=115200, duration=10.0):
    """
    Reads raw packet data from serial for a given duration.
    """
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"✅ Connected to {port} at {baudrate} baud.")
    print("Reading packets... Press Ctrl+C to stop.\n")

    start_time = time.time()
    packet_data = ""

    try:
        while time.time() - start_time < duration:
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore")
                packet_data += line
                print(line.strip())  # optional: live preview

        print("\n⚙️ Processing packets...")
        df = process_packets(packet_data, total_duration=duration)
        print("✅ DataFrame created:\n", df.head())

        # Optionally save
        timestamp = int(time.time())
        df.to_csv(f"packets_{timestamp}.csv", index=False)
        print(f"💾 Saved to packets_{timestamp}.csv")

        return df

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        df = process_packets(packet_data, total_duration=time.time() - start_time)
        df.to_csv("packets_interrupted.csv", index=False)
        print("💾 Saved interrupted capture to packets_interrupted.csv")
        return df

    finally:
        ser.close()
        print("🔌 Serial connection closed.")


if __name__ == "__main__":
    # Example usage: read for 10 seconds
    read_serial_packets(port="/dev/ttyUSB0", baudrate=115200, duration=10.0)
