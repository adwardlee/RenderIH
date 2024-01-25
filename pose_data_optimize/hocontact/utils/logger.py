import logging
import os.path
import time
from termcolor import cprint

logger_inst = None

""" logger = print + log """


def get_logger():
    global logger_inst
    if not logger_inst:
        logger_inst = Logger()
    return logger_inst


class Logger(object):
    def __init__(self):
        self.id = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        self.initialize_done = False

    def initialize(self, save_folder=None, log_name="traineval"):
        if self.initialize_done:
            return self
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if save_folder is None:
            save_folder = "logs"

        os.makedirs(save_folder, exist_ok=True)
        file_path = os.path.join(save_folder, f"{log_name}.log")
        fhandler = logging.FileHandler(file_path, mode="w")
        fhandler.setLevel(self.logger.level)
        self.logger.addHandler(fhandler)

        self.initialize_done = True
        return self

    def info(self, msg, color=None):
        cprint(msg, color=color)
        if self.initialize_done:
            self.logger.info(msg)
        return msg

    def warn(self, msg, color="yellow"):
        cprint(msg, color=color, attrs=["bold"])
        if self.initialize_done:
            self.logger.warning(msg)
        return msg

    def error(self, msg, color="red"):
        cprint(msg, color=color, on_color="on_white", attrs=["bold", "reverse"])
        if self.initialize_done:
            self.logger.error(msg)
        return msg

    def getId(self):
        return self.id


logger = get_logger()
