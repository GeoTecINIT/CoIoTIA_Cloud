from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from model.UseCase import UseCaseRead

class UseCaseService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def get_use_cases(self, user_uid):
        query = "SELECT * FROM use_cases WHERE user_id = %s"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (user_uid,))
                result = await cur.fetchall()
                return [UseCaseRead(**row) for row in result]
 

    async def create_use_case(self, use_case, user_uid):
        query = """
            INSERT INTO use_cases (name, description, keywords, clustering_method, domain_id, user_id)
            VALUES(%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        query,
                        (use_case.name, use_case.description, use_case.keywords, use_case.clustering_method, use_case.domain_id, user_uid)
                    )
                    row = await cur.fetchone()
                    return row[0]