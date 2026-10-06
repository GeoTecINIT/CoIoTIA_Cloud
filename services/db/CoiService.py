from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
import json

from model.Coi import Coi, Centroid

class CoiService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_cois_use_case(self, geometries, properties, use_case_id):
        query = """
            INSERT INTO coi (name, geom, use_case_id)
            VALUES(
                %(name)s,
                ST_SetSRID(
                    ST_GeomFromGeoJSON(%(geom)s),
                    4326
                ),
                %(use_case_id)s
            )
        """

        params = [
            {
                "name" : p["name"],
                "geom" : json.dumps(geom),
                "use_case_id" : use_case_id
            }
            for geom, p in zip(geometries, properties)
        ]

        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(query, params)


    async def list_cois_use_case(self, use_case_id):
        query = "SELECT id, name, ST_AsGeoJSON(geom)::json AS geom, silhouette_score, use_case_id, fog_id FROM coi WHERE use_case_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (use_case_id,))
                result = await cur.fetchall()
                return [Coi(**row) for row in result]


    async def get_use_case_centroid(self, use_case_id):
        query = "SELECT ST_Y(ST_Centroid(ST_Collect(geom))) AS lat, ST_X(ST_Centroid(ST_Collect(geom))) AS lon FROM coi WHERE use_case_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (use_case_id,))
                result = await cur.fetchone()
                return Centroid(**result)


    async def set_cois_of_devices_use_case(self, use_case_id):
        query = """
            UPDATE devices AS d
            SET coi_id = c.id
            FROM coi AS c
            WHERE d.use_case_id = %s
            AND ST_Within(
                    ST_SetSRID(ST_MakePoint(d.lon, d.lat), 4326),
                    c.geom
                )
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (use_case_id, ))
