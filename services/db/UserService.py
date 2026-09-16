from psycopg_pool import AsyncConnectionPool

class UserService:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool