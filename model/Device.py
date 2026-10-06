from pydantic import BaseModel
from datetime import datetime

class Device(BaseModel):
    mac: str
    name: str
    virtual: bool
    mobile: bool
    federated: bool
    analysis_type: int
    data_type: int
    lat: float | None
    lon: float | None
    use_case_id: int | None
    silhouette_score: float | None = None
    status: str = "stopped"

class DeviceCreate(Device):
    pass

class DeviceRead(Device):
    user_id: str
    coi_id: int | None
    last_connection: datetime | None
    last_value: float | None
    total_ram: float | None
    used_ram: float | None