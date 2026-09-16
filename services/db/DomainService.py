from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from model.Domain import Domain

class DomainService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def get_domains(self):
        query = "SELECT * FROM domains"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query)
                result = await cur.fetchall()
                return [Domain(**row) for row in result]