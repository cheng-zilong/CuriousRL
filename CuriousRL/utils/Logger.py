from loguru import logger as _logger
import os
import sys
import json

_logger.remove()
_logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <red>{level}</red>: {message}")

class Logger(object):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    DISABLED = 50
    MIN_LEVEL = 0
    IS_SAVE_JSON = True
    def __init__(self):
        self.logger_id = -100
        
    def logger_init(self, folder_name):
        if self.logger_id == -100:
            self.folder_name = folder_name
            self.log_path = os.path.join("logs", folder_name, "_main.log")
            self.logger_id = _logger.add(self.log_path, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <red>{level}</red>: {message}")
            _logger.info("[+] Create logger file folder \"" + folder_name + "\"!")
        else:
            raise Exception("logger init already done!")

    def logger_destroy(self):
        if self.logger_id == -100:
            raise Exception("There is no existing logger!")
        _logger.remove(self.logger_id)
        self.logger_id = -100

    def set_save_json(self, is_save = False):
        self.IS_SAVE_JSON = False

    def save_to_json(self, **kwargs):
        if self.logger_id != -100 and self.IS_SAVE_JSON:
            data_str = json.dumps(kwargs)
            with open(os.path.join("logs", self.folder_name, "_main.json"), 'a') as file_object:
                file_object.write(data_str + "\n")

    def set_level(self, level):
        """
        Set logging threshold on current logger.
        """
        self.MIN_LEVEL = level

    def debug(self, *args):
        if self.MIN_LEVEL <= self.DEBUG:
            _logger.debug(*args)

    def info(self, *args):
        if self.MIN_LEVEL <= self.INFO:
            _logger.info(*args)

    def warn(self, *args):
        if self.MIN_LEVEL <= self.WARN:
            _logger.warning(*args)

    def error(self, *args):
        if self.MIN_LEVEL <= self.ERROR:
            _logger.error(*args)

# Global logger
logger = Logger() 