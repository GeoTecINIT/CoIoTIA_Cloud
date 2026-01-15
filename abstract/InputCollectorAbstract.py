from abc import ABC, abstractmethod

class InputCollectorAbstract(ABC):
    @abstractmethod
    def collect_input(self):
        pass