from asyncio_mqtt import Client
import asyncio
import time


class MQTTService:
    def __init__(self, broker, port, event_queue, logger, fog_devices):
        self.client = Client(broker, port=port)
        self.event_queue = event_queue
        self.logger = logger
        self.fog_devices = fog_devices
        self._task = None

    async def connect(self):
        await self.client.connect()
        await self.client.subscribe("coiotia/fog", qos=0)
        self.logger.info("Connected to MQTT broker")
        self._task = asyncio.create_task(self._listen())

    async def _listen(self):
        async with self.client.messages() as messages:
            async for message in messages:
                await self.handle_message(message)

    async def disconnect(self):
        if self._task:
            self._task.cancel()
        await self.client.disconnect()

    async def publish(self, topic: str, message: str):
        await self.client.publish(topic, message)

    async def subscribe(self, topic: str):
        await self.client.subscribe(topic)

    async def handle_message(self, message):
        content = message.payload.decode()
        split_content = content.split(";")

        code = split_content[0]
        content_dict = dict(item.split(":") for item in content.split(";")[1:])

        fog_info = self.fog_devices.get(content_dict["NAME"])
        if fog_info is None:
            return

        if code == "ONLINE":
            fog_info["status"] = 1
            fog_info["last_updated"] = time.time()
            self.logger.info(f"Device {content_dict['NAME']} is online at {content_dict['IP']}")
        elif code == "OFFLINE":
            device = content_dict["NAME"]
            fog_info["status"] = 0
            self.logger.info(f"Device {device} is offline")
        else:
            fog_info["last_updated"] = time.time()
            fog_info["status"] = 1
            fog_info["cpu"] = content_dict.get("CPU")
            fog_info["ram"] = content_dict.get("RAM")
            fog_info["disk"] = content_dict.get("Disk")
            self.logger.info(f"Received metrics from {content_dict['NAME']}")

        await self.event_queue.put(
            {
                device: {
                    "id": data["id"],
                    "ip": data["ip"],
                    "status": data["status"],
                }
                for device, data in self.fog_devices.items()
            }
        )
