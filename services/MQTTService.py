from aiomqtt import Client
import asyncio
import time


class MQTTService:
    def __init__(self, broker, port, event_queue, logger, fog_service):
        self.broker = broker
        self.port = port

        self.event_queue = event_queue
        self.logger = logger
        self.fog_service = fog_service

        self.client = None
        self.__task = None

    async def connect(self):
        self.__task = asyncio.create_task(self.__listen())
        self.logger.info("Listener task created")

    async def disconnect(self):
        if self.__task:
            self.__task.cancel()
            try:
                await self.__task
            except asyncio.CancelledError:
                pass
            self.logger.info("Disconnected from MQTT broker")

    async def __listen(self):
        try:
            async with Client(self.broker, self.port) as client:
                self.client = client
                await client.subscribe("coiotia/fog", qos=0)
                self.logger.info("Connected and subscribed")
                async for message in client.messages:
                    await self.__handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"Listener error: {e}")

    async def publish(self, topic: str, message: str):
        if self.client is None:
            self.logger.warning("MQTT client not connected")
            return
        await self.client.publish(topic, message)

    async def subscribe(self, topic: str):
        if self.client is None:
            self.logger.warning("MQTT client not connected")
            return
        await self.client.subscribe(topic)

    async def __handle_message(self, message):
        content = message.payload.decode()
        split_content = content.split(";")

        code = split_content[0]
        content_dict = dict(item.split(":") for item in content.split(";")[1:])

        if code == "ONLINE":
            await self.__online(content_dict)
        elif code == "OFFLINE":
            await self.__offline(content_dict)
        else:
            await self.__status(content_dict)

        # await self.event_queue.put(
        #     {
        #         device: {
        #             "id": data["id"],
        #             "ip": data["ip"],
        #             "status": data["status"],
        #         }
        #         for device, data in self.fog_devices.items()
        #     }
        # )


    async def __online(self, content_dict):
        name = content_dict.get('NAME')
        await self.fog_service.set_status("running", name)
        self.logger.info(f"Device {name} is online")


    async def __offline(self, content_dict):
        name = content_dict.get('NAME')
        await self.fog_service.set_status("stopped", name)
        self.logger.info(f"Device {name} is offline")
        

    async def __status(self, content_dict):
        await self.fog_service.set_vitals(content_dict.get('NAME'), content_dict.get('CPU'), content_dict.get('RAM'), content_dict.get('Disk'))
        
    
