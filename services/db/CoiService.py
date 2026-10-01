from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
import json

from model.Coi import Coi

class CoiService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_cois_use_case(self, geometries, use_case_id):
        query = """
            INSERT INTO coi (geom, use_case_id)
            VALUES(
                ST_SetSRID(
                    ST_GeomFromGeoJSON(%(geom)s),
                    4326
                ),
                %(use_case_id)s
            )
        """

        params = [
            {
                "geom" : json.dumps(geom),
                "use_case_id" : use_case_id
            }
            for geom in geometries
        ]

        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(query, params)


    async def list_cois_use_case(self, use_case_id):
        query = "SELECT id, ST_AsGeoJSON(geom)::json AS geom, silhouette_score, use_case_id, fog_id FROM coi WHERE use_case_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (use_case_id,))
                result = await cur.fetchall()
                return [Coi(**row) for row in result]
