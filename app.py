import cv2
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import threading
import time
import os
from playsound import playsound
import winsound

try:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
except ImportError:
    toaster = None

app = Flask(__name__)


ESP32_CAM_IP = "10.210.138.123" 
CAMERA_STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"
ALARM_FILE = "alarm.mp3" 


latest_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "smoke_detected": 0,
    "human_count": 0 
}
is_alarm_playing = False 
lock = threading.Lock()
processed_frame = None


def send_desktop_notification(title, message):
    if toaster:
        try:
            toaster.show_toast(title, message, duration=5, threaded=True)
        except Exception as e:
            print(f"Notification Error: {e}")
    else:
        print(f"⚠️ {title}: {message} (Install win10toast for popups)")


def play_alarm():
    if os.path.exists(ALARM_FILE):
        try:
            # block=False দিলে কোড আটকে যাবে না
            playsound(ALARM_FILE, block=False)
        except Exception as e:
            print(f"Sound Error: {e}")
    else:
        print(f"❌ {ALARM_FILE} not found")

print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt") 
print("Model loaded successfully.")

@app.route('/api/sensor-data', methods=['POST'])
def receive_data():
    global latest_data, last_alert_time
    try:
        data = request.get_json()
        latest_data['temperature'] = data.get('temperature', 0)
        latest_data['humidity'] = data.get('humidity', 0)
        latest_data['smoke_detected'] = data.get('smoke_detected', 0)

        # ধোঁয়া শনাক্ত হলে অ্যালার্ট
        if latest_data['smoke_detected'] == 1:
            curr_time = time.time()
            if curr_time - last_alert_time > 30: 
                print("🔥 SMOKE DETECTED!")
                send_desktop_notification("Drone Fire Alert", "Warning! Smoke detected in the area.")
                play_alarm()
                last_alert_time = curr_time

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/latest-data', methods=['GET'])
def get_latest_data():
    return jsonify(latest_data)

@app.route('/')
def index():
    return render_template('index.html')

def video_processing_thread():
    global processed_frame, latest_data
    cap = cv2.VideoCapture(CAMERA_STREAM_URL)
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(CAMERA_STREAM_URL)
            continue
        
        results = model(frame, classes=[0], conf=0.25, verbose=False)
        with lock:
            latest_data['human_count'] = len(results[0].boxes)
            processed_frame = results[0].plot() 

def generate_frames():
    global processed_frame
    while True:
        with lock:
            if processed_frame is None: continue
            suc, encoded_image = cv2.imencode('.jpg', processed_frame)
            if not suc: continue
            frame_bytes = encoded_image.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) 

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=video_processing_thread)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
