from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from model.Type import AnalysisType, DataType

class TypeService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def get_analysis_types(self):
        query = "SELECT * FROM analysis_types"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query)
                result = await cur.fetchall()
                return [AnalysisType(**row) for row in result]

    async def get_data_types(self):
            query = "SELECT * FROM data_types"
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query)
                    result = await cur.fetchall()
                    return [DataType(**row) for row in result]