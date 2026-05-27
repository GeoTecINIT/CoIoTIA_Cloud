from aiomqtt import Client
import asyncio
import time

class MQTTService:
    def __init__(self, broker, port, event_queue, logger, fog_devices):
        self.broker = broker
        self.port = port

        self.event_queue = event_queue
        self.logger = logger
        self.fog_devices = fog_devices
        self._task = None

        self.client = None
        self.client_cm = None
        self.__task = None

    async def connect(self):
        self.client_cm = Client(self.broker, self.port)
        self.client = await self.client_cm.__aenter__()
        await self.client.subscribe("coiotia/fog", qos=0)
        self.logger.info("Connected to MQTT broker")
        self.__task = asyncio.create_task(self.__listen())

    async def disconnect(self):
        if self.__task:
            self.__task.cancel()
            try:
                await self.__task
            except asyncio.CancelledError:
                pass
        if self.client_cm:
            await self.client_cm.__aexit__(None, None, None)
            self.logger.info("Disconnected from MQTT broker")

    async def _listen(self):
        async with self.client.messages as messages:
            async for message in messages:
                await self.__handle_message(message)

    async def publish(self, topic: str, message: str):
        await self.client.publish(topic, message)

    async def subscribe(self, topic: str):
        await self.client.subscribe(topic)

    async def __handle_message(self, message):
        content = message.payload.decode()
        split_content = content.split(";")

        code = split_content[0]
        content_dict = dict(item.split(":") for item in content.split(";")[1:])

        fog_info = self.fog_devices.get(content_dict["NAME"])

        if code == "ONLINE":
            self.__online(fog_info, content_dict)
        elif code == "OFFLINE":
            self.__offline(fog_info, content_dict)
        else:
            self.__status(fog_info, content_dict)
            
        await self.event_queue.put(
            {
                device: {
                    "id" : data["id"],
                    "ip" : data["ip"],
                    "status" : data["status"]
                } for device, data in self.fog_devices.items()
            }
        )


    def __online(self, fog_info, content_dict):
        fog_info["status"] = 1
        fog_info["last_updated"] = time.time()
        self.logger.info(f"Device {content_dict['NAME']} is online at {content_dict['IP']}")

    def __offline(self, fog_info, content_dict):
        device = content_dict["NAME"]
        fog_info["status"] = 0
        self.logger.info(f"Device {device} is offline")

    def __status(self, fog_info, content_dict):
        fog_info["last_updated"] = time.time()
        fog_info["status"] = 1
        fog_info["cpu"] = content_dict.get("CPU")
        fog_info["ram"] = content_dict.get("RAM")
        fog_info["disk"] = content_dict.get("Disk")
        self.logger.info(f"Received metrics from {content_dict['NAME']}")

    
