import paho.mqtt.client as mqtt

import firebase_utils

class MQTTConnector:
    def __init__(self, broker, port, announcer, logger):
        self.mqtt_client = mqtt.Client()
        self.broker = broker
        self.port = port
        self.announcer = announcer
        self.logger = logger

    def connect(self):
        try:
            self.mqtt_client.connect(self.broker, self.port)
            self.mqtt_client.loop_start()
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_message = self.on_message
        except Exception as e:
            print(e)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            fog_devices, fog_id_name = firebase_utils.get_fog()
            client.subscribe("coiotia/fog", qos=0)
            logger.info("Connected to MQTT broker")
        else:
            logger.error("Connection failed")
            
    def on_message(client, userdata, message):
        content = message.payload.decode()
        split_content = content.split(";")
        code = split_content[0]
        content_dict = dict(item.split(":") for item in content.split(";")[1:])
        fog_info = fog_devices.get(content_dict["NAME"])
        if code == "ONLINE":
            fog_info["status"] = 1
            fog_info["last_updated"] = time.time()
            logger.info(f"Device {content_dict['NAME']} is online at {content_dict['IP']}")
        elif code == "OFFLINE":
            device = content_dict["NAME"]
            fog_info["status"] = 0
            logger.info(f"Device {device} is offline")
        else:
            fog_info["last_updated"] = time.time()
            fog_info["status"] = 1
            fog_info["cpu"] = content_dict.get("CPU")
            fog_info["ram"] = content_dict.get("RAM")
            fog_info["disk"] = content_dict.get("Disk")
            logger.info(f"Received metrics from {content_dict['NAME']}")
        announcer.set(
            {
                device: {
                    "id" : data["id"],
                    "ip" : data["ip"],
                    "status" : data["status"]
                } for device, data in fog_devices.items()
            }
        )