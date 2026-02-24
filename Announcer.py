import queue

class Announcer:
    def __init__(self):
        self.queue = queue.Queue(maxsize=5)

    def set(self, value: dict) -> None:
        self.queue.put(value)

    def get(self) -> dict:
        return self.queue.get()