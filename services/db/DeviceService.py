from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

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
