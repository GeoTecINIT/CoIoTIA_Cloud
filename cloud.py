from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import paho.mqtt.client as mqtt
import os
import time
import threading
import requests

import firebase_utils

app = Flask(__name__)
CORS(app)

app.config['MQTT_BROKER'] = "150.128.89.87"
app.config['MQTT_PORT'] = 1883

fog_devices = {}

mqtt_client = mqtt.Client()

def check_online():
    current_time = time.time()
    for device, data in fog_devices.items():
        if data["last_updated"] is not None and current_time - data["last_updated"] > 60:
            data["status"] = 0
            print(f"Device {device} is offline due to timeout")

def connect_mqtt():
    try:
        mqtt_client.connect(app.config['MQTT_BROKER'], app.config['MQTT_PORT'])
        mqtt_client.loop_start()
    except Exception as e:
        print(e)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        global fog_devices
        fog_devices = firebase_utils.get_fog()
        client.subscribe("coiotia/fog", qos=0)
        thread = threading.Thread(target=check_online)
        thread.daemon = True
        thread.start()
        print("Connected to broker")
    else:
        print("Connection failed")
        
def on_message(client, userdata, message):
    content = message.payload.decode()
    split_content = content.split(";")
    code = split_content[0]
    content_dict = dict(item.split(":") for item in content.split(";")[1:])
    fog_info = fog_devices.get(content_dict["NAME"])
    if code == "ONLINE":
        fog_info["status"] = 1
        fog_info["last_updated"] = time.time()
        print(f"Device {content_dict['NAME']} is online at {content_dict['IP']}")
    elif code == "OFFLINE":
        device = content_dict["NAME"]
        fog_info["status"] = 0
        print(f"Device {device} is offline")
    else:
        fog_info["last_updated"] = time.time()
        fog_info["status"] = 1
        fog_info["cpu"] = content_dict.get("CPU")
        fog_info["ram"] = content_dict.get("RAM")
        fog_info["disk"] = content_dict.get("Disk")
        print(f"Received metrics from {content_dict['NAME']}")


mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message


############################ CLOUD SPECIFIC ROUTES ###################################

@app.route("/getDevices", methods=['GET'])
def get_devices():
    print(fog_devices)
    devices = {
        device : {
            "id" : data["id"],
            "ip" : data["ip"],
            "status" : data["status"]
        } for device, data in fog_devices.items()
    }
    return jsonify(devices)

@app.route("/getMetrics", methods=['GET'])
def get_metrics():
    device = request.args.get("device")
    if not device or device not in fog_devices:
        return jsonify({"error": "Invalid or missing device"}), 400
    fog_info = fog_devices[device]
    metrics = {
        "CPU": fog_info.get("cpu"),
        "RAM": fog_info.get("ram"),
        "Disk": fog_info.get("disk")
    }
    return jsonify(metrics), 200



############################# MODEL MANAGEMENT ###################################

@app.route("/list", methods=['POST'])
def list_models():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/list", data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/upload", methods=['POST'])
def upload_model():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/upload", files=request.files, data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/update", methods=['POST'])
def update_model():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/update", files=request.files, data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/delete", methods=['POST'])
def delete_model():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/delete", data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/deploy", methods=['POST'])
def deploy_model():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/deploy", data=request.form)
    return make_response(resp.content, resp.status_code)



######################### VIRTUAL DEVICE MANAGEMENT ###############################
@app.route("/listVirtualDevices", methods=['POST'])
def list_virtual_devices():
    target_ip = request.headers.get('X-Target-IP')
    print("Target IP:", target_ip)
    resp = requests.post(f"http://{target_ip}:5000/listVirtualDevices", data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/createVirtualDevice", methods=['POST'])
def create_virtual_device():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/createVirtualDevice", data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/deleteVirtualDevice", methods=['POST'])
def delete_virtual_device():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/deleteVirtualDevice", data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/startVirtualDevice", methods=['POST'])
def start_virtual_device():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/startVirtualDevice", data=request.form)
    return make_response(resp.content, resp.status_code)

@app.route("/stopVirtualDevice", methods=['POST'])
def stop_virtual_device():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/stopVirtualDevice", data=request.form)
    return make_response(resp.content, resp.status_code)



########################## OTHER ROUTES ####################################

@app.route("/getDeploymentModels", methods=['GET'])
def get_deployment_models():
    deployment = request.args.get("deployment")
    models = []
    for device, data in fog_devices.items():
        resp = requests.get(f"http://{data['ip']}:5000/getDeploymentModels?deployment={deployment}")
        models.extend(resp.json())
    return jsonify(models), 200

@app.route("/getCode", methods=['GET'])
def get_code():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.get(f"http://{target_ip}:5000/getCode", params=request.args)
    return make_response(resp.content, resp.status_code)

@app.route("/getOnlineDevices", methods=['POST'])
def get_online_devices():
    target_ip = request.headers.get('X-Target-IP')
    resp = requests.post(f"http://{target_ip}:5000/getOnlineDevices", data=request.form)
    return make_response(resp.content, resp.status_code)





if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        connect_mqtt()
    app.run(host="0.0.0.0", debug=True)