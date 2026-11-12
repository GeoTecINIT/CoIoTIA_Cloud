from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

import httpx

from model.Device import DeviceRead

class DeviceService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def get_devices(self, user_uid):
        query = "SELECT * FROM devices WHERE user_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (user_uid,))
                result = await cur.fetchall()
                return [DeviceRead(**row) for row in result]

    async def get_devices_of_use_case(self, use_case_id, user_uid):
        query = "SELECT * FROM devices WHERE use_case_id = %s AND user_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (use_case_id, user_uid,))
                result = await cur.fetchall()
                return [DeviceRead(**row) for row in result]

    async def get_device(self, mac, user_uid):
        query = "SELECT d.*, uc.name AS use_case, c.name AS coi FROM devices AS d LEFT JOIN use_cases AS uc ON d.use_case_id = uc.id LEFT JOIN coi AS c ON d.coi_id = c.id WHERE d.mac = %s AND d.user_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (mac, user_uid,))
                result = await cur.fetchone()
                return DeviceRead(**result)
    
    async def create_devices_use_case(self, devices, user_uid):
        query = """
            INSERT INTO devices (mac, name, lat, lon, virtual, mobile, federated, analysis_type, data_type, use_case_id, user_id, silhouette_score, status)
            VALUES(%(mac)s, %(name)s, %(lat)s, %(lon)s, %(virtual)s, %(mobile)s, %(federated)s, %(analysis_type)s, %(data_type)s, %(use_case_id)s, %(user_id)s, %(silhouette_score)s, %(status)s)
            ON CONFLICT (mac)
            DO UPDATE SET
                use_case_id = EXCLUDED.use_case_id,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon
        """
        params = [
            {
                "mac" : device.mac,
                "name" : device.name,
                "lat" : device.lat,
                "lon" : device.lon,
                "virtual" : device.virtual,
                "mobile" : device.mobile,
                "federated" : device.federated,
                "analysis_type" : device.analysis_type,
                "data_type" : device.data_type,
                "use_case_id": device.use_case_id,
                "user_id" : user_uid,
                "silhouette_score" : device.silhouette_score,
                "status" : device.status,
            }
            for device in devices
        ]

        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(query, params)


    async def create_virtual_devices(self, use_case_id, uid, production):
        query = """
                    SELECT d.mac, d.name, d.lat, d.lon, d.use_case_id, d.analysis_type, d.data_type, f.id AS fog_id, f.ip AS fog_ip
                    FROM devices AS d JOIN coi AS c ON d.coi_id = c.id JOIN fog AS f ON c.fog_id = f.id
                    WHERE d.use_case_id = %s AND d.virtual = TRUE
                """
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (use_case_id, ))
                async for row in cur:
                    data = {
                        'mac' : row['mac'],
                        'name' : row['name'],
                        'use_case' : row['use_case_id'],
                        'lat' : row['lat'],
                        'lon' : row['lon'],
                        'analysis_type' : row['analysis_type'],
                        'data_type' : row['data_type']
                    }
                    if production:
                        url = f'https://{row['fog_ip']}/fog/virtual/create'
                    elif int(row['fog_id']) < 5:
                        url = f'https://aiotserver.uji.es/fog/{row['fog_id']}/virtual/create'
                    else:
                        url = 'http://localhost:8001/virtual/create'
                    r = httpx.post(url, headers={'x-user-uid' : uid}, data=data)


    async def set_status(self, mac, status):
        query = "UPDATE devices SET status = %s WHERE mac = %s RETURNING name"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (status, mac))
                result = await cur.fetchone()
                if result:
                    name = result[0]
                    return name
                return None


    async def set_vitals(self, mac, total_ram, used_ram):
        query = "UPDATE devices SET total_ram = %s, used_ram = %s, last_connection = NOW() WHERE mac = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (total_ram, used_ram, mac))


    async def set_position(self, mac, lat, lon):
        query = "UPDATE devices SET lat = %s, lon = %s, last_connection = NOW() WHERE mac = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (lat, lon, mac))


    async def delete_devices_use_case(self, use_case_id):
        query = "DELETE FROM devices WHERE use_case_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (use_case_id, ))
    
