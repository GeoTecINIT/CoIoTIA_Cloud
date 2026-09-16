from psycopg_pool import AsyncConnectionPool

class CoiService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool