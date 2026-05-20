# CoIoTIA_Cloud — Registro de Cambios

## Fecha: 2026-05-20

## Resumen

Se ha añadido soporte para **dispositivos federados (Federated Learning)** en la API del Cloud y se han corregido inconsistencias en los paths de forwarding y en el uso del header `X-Target-IP`.

---

## Archivos Nuevos

### `api/utils.py`

Módulo compartido con utilidades para los routers:

- **`forward_request(path, request, x_target_ip, files=None)`**: Función genérica de proxy que reenvía una petición form-data (+ archivos opcionales) a un nodo Fog en `http://{x_target_ip}/{path}`. Centraliza la lógica de forwarding que antes estaba duplicada en 3 archivos (`models.py`, `virtual_devices.py`, `extra.py`).
- **`build_fog_url(ip, path)`**: Construye URLs para peticiones broadcast (sin `X-Target-IP`) usando el formato `http://{ip}:5000/{path}`. Puerto 5000 fijo, consistente con el comportamiento anterior.

### `api/routes/federated.py`

Nuevo router con prefijo `/federated` que añade soporte para dispositivos federados:

| Endpoint | Método | Descripción |
|---|---|---|
| `POST /federated/list` | POST | Proxy a un Fog individual. Reenvía form data a `POST {fog_ip}/federated/list`. Requiere header `X-Target-IP`. |
| `GET /federated/listAll` | GET | Agrega dispositivos federados de **todos** los Fog online en paralelo (`asyncio.gather`). Realiza `POST {fog_ip}:5000/federated/list` a cada Fog con `status == 1`. Retorna lista unificada. |

---

## Archivos Modificados

### `app.py`

- **Línea 3**: Añadido import de `federated` en `from api.routes import models, virtual_devices, cloud, extra, federated`
- **Línea 70**: Añadido registro del router federado: `app.include_router(federated.router, prefix="/federated")`

### `api/routes/models.py`

Cambios de consistencia:

- **Import**: Se reemplazó el `forward_request` local por el compartido de `api.utils`.
- **`POST /deploy`**: Se cambió `x_target_ip` de **query parameter** a **Header `X-Target-IP`** (`Header(...)`). Antes era inconsistente con el resto de endpoints del mismo router.

| Antes | Después |
|---|---|
| `async def deploy_model(request: Request, x_target_ip: str)` (query param) | `async def deploy_model(request: Request, x_target_ip: str = Header(...))` |

- Se eliminó la función `forward_request` local (ahora se usa `api.utils.forward_request`).

### `api/routes/virtual_devices.py`

Cambios sustanciales:

1. **Paths de forwarding corregidos** (de API Flask a API FastAPI):

   | Path antiguo (Flask, puerto 5000) | Path nuevo (FastAPI) |
   |---|---|
   | `listVirtualDevices` | `virtual/list` |
   | `createVirtualDevices` | `virtual/create` |
   | `deleteVirtualDevices` | `virtual/delete` |
   | `startVirtualDevices` | `virtual/start` |
   | `stopVirtualDevices` | `virtual/stop` |

2. **Nuevo endpoint `POST /virtual/online`**: Proxy a Fog `POST /virtual/online`. Requiere header `X-Target-IP`. Retorna dispositivos reales + virtuales + federados.

3. **`POST /virtual/createAll` actualizado con soporte federado**:
   - Nuevos parámetros form: `node_type` (default `"traditional"`), `cluster_id`, `mqtt_broker_host`, `mqtt_broker_port`, `mqtt_network`, `image`, `sensor_api_base_url`, `start`.
   - Cuando `node_type == "federated"`: para cada sensor en cada región, envía un `POST /virtual/create` individual con todos los campos federados (`sensor_id`, `cluster_id`, `mqtt_broker_host`, etc.).
   - Cuando `node_type == "traditional"`: envía `POST /virtual/create` con `user_uid`, `sensors` (JSON), y `virtual_type`.
   - Las peticiones se ejecutan en paralelo con `asyncio.gather(*tasks, return_exceptions=True)`.
   - Se usa `build_fog_url(ip, "virtual/create")` para construir URLs con puerto 5000.

4. **Se eliminó la función `forward_request` local**. Los endpoints proxy ahora usan `httpx.AsyncClient` directamente con `f"http://{x_target_ip}/virtual/{path}"`.

5. **`X-Target-IP` consistente**: Todos los endpoints proxy usan `Header(...)`.

### `api/routes/extra.py`

Cambios de consistencia y corrección:

1. **`GET /getCode`**:
   - Cambiado `x_target_ip` de **query parameter** a **Header `X-Target-IP`**.
   - Corregido el bug de intentar leer form data en un GET request. Ahora se pasa el query string directamente al Fog.
   - Antes: `forward_request("getCode", request, x_target_ip)` que llamaba `await request.form()` en un GET.
   - Ahora: Construye la URL con query string y hace `client.get(url)`.

2. **`POST /getOnlineDevices`**:
   - Cambiado `x_target_ip` de **query parameter** a **Header `X-Target-IP`**.
   - Path de forwarding corregido: de `getOnlineDevices` a `virtual/online`.
   - Se eliminó la función `forward_request` local.
   - Ahora envía form data correctamente al Fog.

3. **`GET /getDeploymentModels`**:
   - Se usa `build_fog_url(data["ip"], "getDeploymentModels")` en vez de URL hardcodeada con `:5000`.
   - Funcionalmente equivalente (mismo puerto 5000), pero centralizado en `api.utils`.

4. **Se eliminó la función `forward_request` local** (3 copias eliminadas en total de `models.py`, `virtual_devices.py`, `extra.py`).

---

## Mapeo Completo de Endpoints Cloud → Fog (Actualizado)

| Cloud Endpoint | Método | Fog Target | IP Source | Notas |
|---|---|---|---|---|
| `/models/list` | POST | `/list` | Header `X-Target-IP` | Sin cambios funcionales |
| `/models/upload` | POST | `/upload` | Header `X-Target-IP` | Sin cambios funcionales |
| `/models/update` | POST | `/update` | Header `X-Target-IP` | Sin cambios funcionales |
| `/models/delete` | POST | `/delete` | Header `X-Target-IP` | Sin cambios funcionales |
| `/models/deploy` | POST | `/deploy` | Header `X-Target-IP` | **Corregido**: era query param |
| `/virtual/createAll` | POST | `/virtual/create` × N | Auto (Firebase) | **Actualizado**: soporte federado, paralelo |
| `/virtual/list` | POST | `/virtual/list` | Header `X-Target-IP` | Path corregido |
| `/virtual/create` | POST | `/virtual/create` | Header `X-Target-IP` | Path corregido |
| `/virtual/delete` | POST | `/virtual/delete` | Header `X-Target-IP` | Path corregido |
| `/virtual/start` | POST | `/virtual/start` | Header `X-Target-IP` | Path corregido |
| `/virtual/stop` | POST | `/virtual/stop` | Header `X-Target-IP` | Path corregido |
| `/virtual/online` | POST | `/virtual/online` | Header `X-Target-IP` | **Nuevo** |
| `/federated/list` | POST | `/federated/list` | Header `X-Target-IP` | **Nuevo** |
| `/federated/listAll` | GET | `/federated/list` × N | Auto (todos los Fog online) | **Nuevo** |
| `/getDevices` | GET | Local (SSE) | N/A | Sin cambios |
| `/getMetrics` | GET | Local (dict) | N/A | Sin cambios |
| `/determineRegion` | POST | Local (GeoPandas) | N/A | Sin cambios |
| `/getDeploymentModels` | GET | `/getDeploymentModels` × N | Auto (todos los Fog) | URL centralizada |
| `/getCode` | GET | `/getCode` | Header `X-Target-IP` | **Corregido**: era query param, GET fixeado |
| `/getOnlineDevices` | POST | `/virtual/online` | Header `X-Target-IP` | **Corregido**: era query param, path actualizado |

---

## Detalle del Flujo Federado en `createAll`

```
POST /virtual/createAll
  Form data:
    user: str (requerido)
    domain: str (requerido)
    sensors: JSON str (requerido)
      - [{id, lat, lon, virtual, mobile, region, analysis_type?, data_type?, name?}]
    node_type: str (default "traditional")
      - "traditional" | "federated"
    virtual_type: str (default = node_type)
    cluster_id: str (opcional)
    mqtt_broker_host: str (opcional)
    mqtt_broker_port: str (opcional)
    mqtt_network: str (opcional)
    image: str (opcional)
    sensor_api_base_url: str (opcional)
    start: str (default "false")

  Lógica:
    1. group_sensors_by_region(json.loads(sensors))
    2. get_fog_of_regions(user, domain) → {region_name: fog_doc_id}
    3. Resolver fog_doc_id → fog_name → IP
    4. Si node_type == "federated":
       Para cada sensor en cada región:
         POST http://{fog_ip}:5000/virtual/create con:
           user_uid, name, analysis_type, data_type, lat, lon,
           node_type=federated, sensor_id, cluster_id, mqtt_broker_host,
           mqtt_broker_port, mqtt_network, image, sensor_api_base_url, start
    5. Si node_type == "traditional":
       Para cada región:
         POST http://{fog_ip}:5000/virtual/create con:
           user_uid, sensors (JSON), virtual_type
    6. asyncio.gather para ejecutar en paralelo
    7. Retornar {"message": "Virtual devices creation initiated"}
```

---

## Notas sobre Compatibilidad con el Frontend

### Cambios aplicados en CoIoTIA_Frontend

Los siguientes paths fueron actualizados en el Frontend para coincidir con la API del Cloud:

| Archivo | Path antiguo | Path nuevo |
|---|---|---|
| `devices.component.ts` | `/listVirtualDevices` | `/virtual/list` |
| `devices.component.ts` | `/createVirtualDevice` | `/virtual/create` |
| `devices.component.ts` | `/deleteVirtualDevice` | `/virtual/delete` |
| `devices.component.ts` | `/startVirtualDevice` | `/virtual/start` |
| `devices.component.ts` | `/stopVirtualDevice` | `/virtual/stop` |
| `models.component.ts` | `/listVirtualDevices` | `/virtual/list` |
| `map.component.ts` | `/createAllVirtualDevices` | `/virtual/createAll` |

### Sin Cambios en el Frontend

- Los endpoints locales del Cloud (`/getDevices`, `/getMetrics`, `/determineRegion`) no cambian.
- `POST /models/*` no cambia desde la perspectiva del Frontend (el path es el mismo).
- `/getOnlineDevices` ya usaba `X-Target-IP` como Header, no necesita cambio.
- `/getCode` ya usaba `X-Target-IP` como Header, no necesita cambio.

### Nuevos endpoints disponibles (no integrados aún en el Frontend)

- `POST /federated/list` — Listar dispositivos federados de un Fog
- `GET /federated/listAll` — Listar todos los dispositivos federados de todos los Fogs
- `POST /virtual/online` — Obtener dispositivos online (reales+virtuales+federados) de un Fog
- `POST /virtual/createAll` — Ahora acepta `node_type=federated` y campos federados adicionales

---

## Archivos No Modificados

| Archivo | Razón |
|---|---|
| `api/routes/cloud.py` | Sin cambios. Endpoints locales (SSE, métricas, regiones) no necesitan modificación. |
| `services/MQTTService.py` | Sin cambios. No se suscribe a topics federados (decisión: solo REST proxy por ahora). |
| `services/Logger.py` | Sin cambios. |
| `services/Announcer.py` | Sin cambios (ya no se usa). |
| `firebase_utils.py` | Sin cambios. |
| `utils.py` | Sin cambios. |
| `requirements.txt` | Sin cambios (no se añadieron nuevas dependencias). |