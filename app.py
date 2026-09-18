from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import models, virtual_devices, cloud, extra, federated, use_cases, domains, devices, types
from contextlib import asynccontextmanager
import asyncio
from asyncio import Queue
import time
import os

from psycopg_pool import AsyncConnectionPool

from dotenv import load_dotenv
load_dotenv()

from services.Logger import Logger
from services.MQTTService import MQTTService
from services.FirebaseService import FirebaseService

from services.db.UseCaseService import UseCaseService
from services.db.UserService import UserService
from services.db.CoiService import CoiService
from services.db.FogService import FogService
from services.db.DomainService import DomainService
from services.db.DeviceService import DeviceService
from services.db.TypeService import TypeService


async def check_online_task(app: FastAPI):
    while True:
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
    app.state.firebase = FirebaseService()

    app.state.logger.info("===================================")
    app.state.logger.info("      STARTING COIOTIA CLOUD       ")
    app.state.logger.info("===================================")
    
    pool = AsyncConnectionPool(
        conninfo=os.getenv("DB_URL"),
        min_size=4,
        max_size=20,
        open=False
    )
    await pool.open()

    app.state.use_case_service = UseCaseService(pool)
    app.state.coi_service = CoiService(pool)
    app.state.fog_service = FogService(pool)
    app.state.user_service = UserService(pool)
    app.state.domain_service = DomainService(pool)
    app.state.device_service = DeviceService(pool)
    app.state.types_service = TypeService(pool)
    
    app.state.event_queue = Queue(maxsize=5)
    app.state.fog_devices, app.state.fog_id_name = app.state.firebase.get_fog()
    app.state.logger.info(f"Loaded {len(app.state.fog_devices)} fog devices from Firestore")
    app.state.mqtt = MQTTService(os.getenv("MQTT_BROKER", "aiotserver.uji.es"), int(os.getenv("MQTT_PORT", 1883)), app.state.event_queue, app.state.logger, app.state.fog_devices)

    await app.state.mqtt.connect()
    app.state.check_online_task = asyncio.create_task(check_online_task(app))
    
    yield

    pool.close()
    
    task = app.state.check_online_task
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

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
app.include_router(use_cases.router, prefix="/useCase")
app.include_router(domains.router, prefix="/domains")
app.include_router(devices.router, prefix="/devices")
app.include_router(types.router, prefix="/types")