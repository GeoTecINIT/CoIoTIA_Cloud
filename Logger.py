import sys
import logging
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(self, name, app):
        self.logger = logging.getLogger(name)
        self.name = name
        self.app = app
        self.setup_logger()

    def setup_logger(self):
        for h in self.app.logger.handlers:
            self.app.logger.removeHandler(h)

        file_handler = RotatingFileHandler(
            f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        self.logger.addHandler(file_handler)

        if sys.stdout.isatty():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.INFO)