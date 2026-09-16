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
    use_case_id: int | None
    silhouette_score: float | None = None
    status: str = "stopped"

class DeviceCreate(Device):
    pass

class DeviceRead(Device):
    lat: float | None
    lon: float | None
    user_id: str
    last_connection: datetime | None
    last_value: float | None
    cpu: float | None
    memory: float | None