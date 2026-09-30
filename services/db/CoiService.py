from psycopg_pool import AsyncConnectionPool
import json

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

