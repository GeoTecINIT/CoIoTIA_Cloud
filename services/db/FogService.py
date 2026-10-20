from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from model.Fog import Fog, FogVitals

class FogService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def get_fog(self):
        query = "SELECT * FROM fog"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query)
                result = await cur.fetchall()
                return [Fog(**row) for row in result]

    async def get_vitals(self, fog_id):
        query = "SELECT cpu, ram, disk FROM fog WHERE id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (fog_id, ))
                result = await cur.fetchone()
                return FogVitals(**result)

    async def set_vitals(self, name, cpu, ram, disk):
        query = "UPDATE fog SET cpu = %s, ram = %s, disk = %s WHERE name = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (cpu, ram, disk, name))

    async def set_status(self, status, name):
        query = "UPDATE fog SET status = %s WHERE name = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (status, name))

    async def check_edge_in_fog(self, fog_id, mac):
        query = """
            SELECT EXISTS(
                SELECT 1
                FROM fog AS f
                JOIN coi AS c ON f.id = c.fog_id
                JOIN devices AS d ON c.id = d.coi_id
                WHERE f.id = %s
                AND d.mac = %s
            )

        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (fog_id, mac))
                result = await cur.fetchone()
                return result[0]