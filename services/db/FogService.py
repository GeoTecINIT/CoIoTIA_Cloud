from psycopg_pool import AsyncConnectionPool

class FogService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool