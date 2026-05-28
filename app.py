from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import models, virtual_devices, cloud, extra, federated
from contextlib import asynccontextmanager
import asyncio
from asyncio import Queue
import time
import os

from services.Logger import Logger
from services.MQTTService import MQTTService

import firebase_utils

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

async def check_online_task(app: FastAPI):
    current_time = time.time()
    for device, data in app.state.fog_devices.items():
        if data["last_updated"] is not None and current_time - data["last_updated"] > 60:
            data["status"] = 0
            await app.state.event_queue.put(
                {
                    device: {
                        "id" : data["id"],
                        "ip" : data["ip"],
                        "status" : data["status"]
                    } for device, data in app.state.fog_devices.items()
                }
            )
            app.state.logger.info(f"Device {device} is offline due to timeout")
    
    await asyncio.sleep(60)  # Check every 60 seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = Logger("CoIoTIA_Cloud")

    app.state.logger.info("===================================")
    app.state.logger.info("      STARTING COIOTIA CLOUD       ")
    app.state.logger.info("===================================")
    
    app.state.event_queue = Queue(maxsize=5)
    app.state.fog_devices, app.state.fog_id_name = firebase_utils.get_fog()
    app.state.mqtt = MQTTService(MQTT_BROKER, MQTT_PORT, app.state.event_queue, app.state.logger, app.state.fog_devices)
    app.state.check_online_task = asyncio.create_task(check_online_task(app))
    await app.state.mqtt.connect()

    yield
    
    await app.state.mqtt.disconnect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/models")
app.include_router(virtual_devices.router, prefix="/virtual")
app.include_router(cloud.router)
app.include_router(extra.router)
app.include_router(federated.router, prefix="/federated")