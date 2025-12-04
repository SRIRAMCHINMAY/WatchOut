#!/usr/bin/env python3
"""
MQTT Receiver for ESP32 WiFi Packet Analysis - FIXED VERSION
Receives packets and handshakes from ESP32 and feeds them to ML model
"""

import json
import paho.mqtt.client as mqtt
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
import time
import os
import pickle
import sys

# ==================== CONFIGURATION ====================
MQTT_BROKER = "broker.hivemq.com"  # Change this to your MQTT broker IP
MQTT_PORT = 1883
MQTT_USERNAME = ""
MQTT_PASSWORD = ""

# Topics
TOPIC_STATUS = "esp32/status"
TOPIC_PACKETS = "esp32/packets"
TOPIC_PACKET_STREAM = "esp32/packets/stream"
TOPIC_HANDSHAKE = "esp32/handshake"
TOPIC_HANDSHAKE_PROGRESS = "esp32/handshake/progress"
TOPIC_NETWORKS = "esp32/networks"
TOPIC_DEAUTH = "esp32/deauth"
TOPIC_CONTROL = "esp32/control"

# Data storage
packet_queue = queue.Queue()
handshake_data = {}
network_list = []
all_packets = []

# Create output directory
OUTPUT_DIR = "esp32_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== ML MODEL ====================
class WiFiMLModel:
    """ML Model for WiFi intrusion detection"""
    
    def __init__(self, model_path='intrusion_detection_model.pkl'):
        print("[ML] Initializing ML Model...")
        
        try:
            with open(model_path, 'rb') as f:
                self.loaded_model = pickle.load(f)
            print("[ML] ✓ Model loaded successfully!")
            
            # Get model info
            if hasattr(self.loaded_model, 'n_features_in_'):
                self.expected_features = self.loaded_model.n_features_in_
                print(f"[ML]   Model expects {self.expected_features} features")
            else:
                self.expected_features = 11  # Default based on your dataset
                
            if hasattr(self.loaded_model, 'classes_'):
                print(f"[ML]   Classes: {self.loaded_model.classes_}")
                
        except FileNotFoundError:
            print(f"[ML] ✗ Error: {model_path} not found!")
            raise
        except Exception as e:
            print(f"[ML] ✗ Error loading model: {e}")
            raise
        
        # Define feature columns - MATCHING YOUR MODEL'S 7 FEATURES
        self.feature_columns = [
            'frame.len',
            'frame.cap_len',
            'radiotap.dbm_antsignal',
            'radiotap.channel.freq',
            'frame.time_delta',
            'frame.time_relative',
            'radiotap.present.channel'
        ]
        
        print(f"[ML] Using {len(self.feature_columns)} features:")
        for i, feat in enumerate(self.feature_columns, 1):
            print(f"      {i}. {feat}")
        print("[ML] ✓ Model initialized successfully!\n")
    
    def mac_to_int(self, mac_str):
        """Convert MAC address to integer"""
        try:
            # Remove colons and convert hex to int
            mac_clean = mac_str.replace(':', '')
            return int(mac_clean, 16)
        except:
            return 0
    
    def preprocess_packet(self, packet):
        """Convert ESP32 packet data to model features (7 features only)"""
        
        # Calculate time delta (time since last packet)
        # For simplicity, using a running average or 0 for first packet
        if not hasattr(self, '_last_timestamp'):
            self._last_timestamp = packet.get('timestamp', 0)
            time_delta = 0
        else:
            current_time = packet.get('timestamp', 0)
            time_delta = (current_time - self._last_timestamp) / 1000000.0  # Convert to seconds
            self._last_timestamp = current_time
        
        # Map ESP32 packet fields to the 7 required features
        features = {
            # Frame length features
            'frame.len': packet.get('length', 0),
            'frame.cap_len': packet.get('length', 0),
            
            # Signal strength
            'radiotap.dbm_antsignal': packet.get('rssi', -50),
            
            # Channel frequency
            'radiotap.channel.freq': self._channel_to_freq(packet.get('channel', 1)),
            
            # Timing features
            'frame.time_delta': time_delta,
            'frame.time_relative': packet.get('timestamp', 0) / 1000000.0,  # Convert to seconds
            
            # Channel present flag
            'radiotap.present.channel': 1 if packet.get('channel') else 0,
        }
        
        return features
    
    def _channel_to_freq(self, channel):
        """Convert WiFi channel number to frequency in MHz"""
        if channel == 0:
            return 0
        elif 1 <= channel <= 13:
            # 2.4 GHz band
            return 2412 + (channel - 1) * 5
        elif channel == 14:
            return 2484
        elif 36 <= channel <= 165:
            # 5 GHz band
            return 5000 + channel * 5
        else:
            return 2437  # Default to channel 6
    
    def preprocess_batch(self, packets):
        """Preprocess a batch of packets into a DataFrame"""
        feature_list = []
        for pkt in packets:
            features = self.preprocess_packet(pkt)
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list)[self.feature_columns]
        return df
    
    def predict(self, packets):
        """Make predictions on packet data"""
        if not packets:
            return []
        
        df = self.preprocess_batch(packets)
        
        try:
            predictions_raw = self.loaded_model.predict(df)
            
            results = []
            for idx, pred in enumerate(predictions_raw):
                # Handle different prediction formats
                if isinstance(pred, (int, np.integer)):
                    # Class label (0, 1, 2, 3, etc.)
                    is_anomaly = bool(pred != 0)  # Assuming 0 is normal
                    confidence = 0.6 if is_anomaly else 0.1
                    attack_type = self._get_attack_type(pred)
                    
                elif isinstance(pred, (float, np.floating)):
                    # Probability
                    is_anomaly = pred > 0.5
                    confidence = float(pred) if is_anomaly else float(1 - pred)
                    attack_type = 'attack' if is_anomaly else 'normal'
                    
                else:
                    # Array of probabilities
                    is_anomaly = pred[1] > 0.5 if len(pred) > 1 else pred[0] > 0.5
                    confidence = float(pred[1]) if len(pred) > 1 else float(pred[0])
                    attack_type = 'attack' if is_anomaly else 'normal'
                
                results.append({
                    'packet_id': packets[idx].get('id'),
                    'anomaly': is_anomaly,
                    'confidence': confidence,
                    'attack_type': attack_type,
                    'prediction': int(pred) if isinstance(pred, (int, np.integer)) else pred,
                    'features': df.iloc[idx].to_dict()
                })
            
            return results
            
        except Exception as e:
            print(f"[ML] ✗ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            
            return [{
                'packet_id': packets[idx].get('id'),
                'anomaly': False,
                'confidence': 0.0,
                'attack_type': 'error',
                'error': str(e)
            } for idx in range(len(packets))]
    
    def _get_attack_type(self, class_label):
        """Map class labels to attack types (adjust based on your model)"""
        attack_mapping = {
            0: 'normal',
            1: 'flooding',
            2: 'injection',
            3: 'impersonation',
            # Add more mappings based on your model's classes
        }
        return attack_mapping.get(int(class_label), f'attack_class_{class_label}')
    
    def analyze_handshake(self, handshake):
        """Analyze captured handshake data"""
        print("\n[ML] Analyzing Handshake...")
        
        features = {
            'has_m1': 1 if 'm1' in handshake else 0,
            'has_m2': 1 if 'm2' in handshake else 0,
            'has_m3': 1 if 'm3' in handshake else 0,
            'has_m4': 1 if 'm4' in handshake else 0,
            'deauth_count': handshake.get('deauth_count', 0),
            'capture_duration': handshake.get('capture_duration_ms', 0),
        }
        
        is_valid = features['has_m1'] and features['has_m2']
        is_attack = features['deauth_count'] > 50
        
        result = {
            'handshake_valid': is_valid,
            'potential_attack': is_attack,
            'attack_type': 'deauth_attack' if is_attack else 'legitimate',
            'confidence': 0.85 if is_attack else 0.15,
            'features': features,
            'recommendation': 'ALERT: Possible deauth attack detected' if is_attack else 'Handshake appears normal'
        }
        
        print(f"[ML] Valid: {result['handshake_valid']}, Attack: {result['potential_attack']}")
        return result


# ==================== MQTT CALLBACKS (API v2) ====================
def on_connect(client, userdata, flags, reason_code, properties):
    """Callback for when client connects to broker - API v2"""
    if reason_code == 0:
        print(f"[MQTT] ✓ Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
        
        topics = [
            (TOPIC_STATUS, 0),
            (TOPIC_PACKETS, 0),
            (TOPIC_PACKET_STREAM, 0),
            (TOPIC_HANDSHAKE, 0),
            (TOPIC_HANDSHAKE_PROGRESS, 0),
            (TOPIC_NETWORKS, 0),
            (TOPIC_DEAUTH, 0),
            (TOPIC_CONTROL, 0)
        ]
        client.subscribe(topics)
        print("[MQTT] ✓ Subscribed to all topics\n")
        print("="*60)
        print("Waiting for data from ESP32...")
        print("="*60 + "\n")
    else:
        print(f"[MQTT] ✗ Connection failed with reason code: {reason_code}")


def on_disconnect(client, userdata, flags, reason_code, properties):
    """Callback for when client disconnects - API v2"""
    print(f"[MQTT] Disconnected with reason code: {reason_code}")


def on_message(client, userdata, msg):
    """Callback for when a message is received"""
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())
    except:
        payload = msg.payload.decode()
        print(f"[DEBUG] Non-JSON on {topic}: {payload}")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if topic == TOPIC_STATUS:
        handle_status(payload, timestamp)
    elif topic == TOPIC_PACKETS:
        handle_packet_summary(payload, timestamp)
    elif topic == TOPIC_PACKET_STREAM:
        handle_packet_stream(payload, timestamp)
    elif topic == TOPIC_HANDSHAKE:
        handle_handshake(payload, timestamp)
    elif topic == TOPIC_HANDSHAKE_PROGRESS:
        handle_handshake_progress(payload, timestamp)
    elif topic == TOPIC_NETWORKS:
        handle_networks(payload, timestamp)
    elif topic == TOPIC_DEAUTH:
        handle_deauth(payload, timestamp)


# ==================== MESSAGE HANDLERS ====================
def handle_status(data, timestamp):
    status = data.get('status', 'unknown')
    print(f"\n[{timestamp}] 📡 STATUS: {status}")
    if 'ip' in data:
        print(f"  └─ Device IP: {data['ip']}")


def handle_packet_summary(data, timestamp):
    total = data.get('total_packets', 0)
    duration = data.get('duration_ms', 0)
    print(f"\n[{timestamp}] 📦 PACKET CAPTURE COMPLETE")
    print(f"  ├─ Total Packets: {total}")
    print(f"  └─ Duration: {duration/1000:.2f}s")


def handle_packet_stream(data, timestamp):
    packets = data.get('packets', [])
    batch_start = data.get('batch_start', 0)
    total = data.get('total_packets', 0)
    
    print(f"\n[{timestamp}] 📨 Received packet batch: {len(packets)} packets")
    
    all_packets.extend(packets)
    
    for pkt in packets:
        packet_queue.put(pkt)
    
    if packets:
        sample = packets[0]
        print(f"  └─ Sample: {sample.get('src')} → {sample.get('dst')} "
              f"(RSSI: {sample.get('rssi')} dBm)")


def handle_handshake(data, timestamp):
    global handshake_data
    handshake_data = data
    
    print(f"\n[{timestamp}] 🔐 HANDSHAKE CAPTURED!")
    print(f"  ├─ AP: {data.get('ap_mac')}")
    print(f"  └─ Client: {data.get('client_mac')}")
    
    if ml_model:
        analysis = ml_model.analyze_handshake(data)
        save_handshake_analysis(data, analysis, timestamp)
        
        if analysis.get('potential_attack'):
            print("\n" + "!"*60)
            print("⚠  SECURITY ALERT: Potential Attack Detected!")
            print("!"*60 + "\n")


def handle_handshake_progress(data, timestamp):
    msg_type = data.get('type')
    if msg_type == 'handshake_message':
        print(f"[{timestamp}] 🔑 EAPOL Message {data.get('message_num')} captured")


def handle_networks(data, timestamp):
    global network_list
    network_list = data.get('networks', [])
    count = len(network_list)
    
    print(f"\n[{timestamp}] 📶 NETWORKS FOUND: {count}")
    for i, net in enumerate(network_list[:10], 1):
        ssid = net.get('ssid', 'Unknown')[:20]
        print(f"  {i}. {ssid} - Ch:{net.get('channel')} - {net.get('rssi')} dBm")


def handle_deauth(data, timestamp):
    print(f"[{timestamp}] 💥 Deauth: {data.get('count')} packets")


# ==================== ML PROCESSING THREAD ====================
def ml_processing_thread(model):
    """Background thread for ML processing"""
    print("[ML] 🚀 Starting ML processing thread...")
    batch_size = 10
    batch = []
    
    while True:
        try:
            packet = packet_queue.get(timeout=1)
            batch.append(packet)
            
            if len(batch) >= batch_size:
                predictions = model.predict(batch)
                anomaly_count = sum(1 for p in predictions if p['anomaly'])
                
                if anomaly_count > 0:
                    print(f"\n[ML] ⚠  {anomaly_count}/{len(batch)} anomalies detected")
                    for pred in predictions:
                        if pred['anomaly']:
                            print(f"  └─ Packet #{pred['packet_id']}: {pred['attack_type']} "
                                  f"(confidence: {pred['confidence']:.2%})")
                else:
                    print(f"[ML] ✓ Batch of {len(batch)} packets - all normal")
                
                save_predictions(batch, predictions)
                batch = []
                
        except queue.Empty:
            if batch:
                predictions = model.predict(batch)
                save_predictions(batch, predictions)
                batch = []
            continue
        except Exception as e:
            print(f"[ML] ✗ Error: {e}")


# ==================== DATA PERSISTENCE ====================
def save_predictions(packets, predictions):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.json")
    
    data = {
        'timestamp': timestamp,
        'packet_count': len(packets),
        'anomaly_count': sum(1 for p in predictions if p['anomaly']),
        'predictions': predictions
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[SAVE] ✗ Error: {e}")


def save_handshake_analysis(handshake, analysis, timestamp):
    ts_str = timestamp.replace(':', '-').replace(' ', '_')
    filename = os.path.join(OUTPUT_DIR, f"handshake_{ts_str}.json")
    
    data = {'timestamp': timestamp, 'handshake': handshake, 'analysis': analysis}
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[SAVE] ✓ Saved to {filename}")
    except Exception as e:
        print(f"[SAVE] ✗ Error: {e}")


# ==================== MAIN ====================
ml_model = None

def main():
    global ml_model
    
    print("="*60)
    print("  ESP32 MQTT RECEIVER & ML PROCESSOR")
    print("="*60)
    print(f"  MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Initialize ML model
    try:
        ml_model = WiFiMLModel()
    except Exception as e:
        print(f"[ERROR] Failed to initialize ML model: {e}")
        print("[INFO] Continuing without ML processing...")
        ml_model = None
    
    # Create MQTT client with CallbackAPIVersion.VERSION2
    client = mqtt.Client(
        client_id="ML_Processor",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    # Set authentication if needed
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    # Start ML processing thread if model loaded
    if ml_model:
        ml_thread = threading.Thread(target=ml_processing_thread, args=(ml_model,), daemon=True)
        ml_thread.start()
    
    # Connect to broker with retry
    print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            print(f"[ERROR] Connection attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print("[INFO] Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("\n[ERROR] Could not connect to MQTT broker.")
                print("[FIX] Check:")
                print("  1. MQTT broker is running (mosquitto, HiveMQ, etc.)")
                print("  2. Broker IP address is correct")
                print("  3. Port 1883 is not blocked by firewall")
                print("  4. ESP32 and this computer are on same network")
                sys.exit(1)
    
    # Start listening
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        client.disconnect()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()