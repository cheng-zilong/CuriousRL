from loguru import logger as _logger
import os
import sys
import json
from pathlib import Path
import shutil

class Logger(object):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    DISABLED = 50
    __min_level = 0
    __is_use_logger = True
    __folder_name = None
    __is_save_json = False

    def __init__(self):
        _logger.remove()
        _logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <red>{level}</red>: {message}")
        from datetime import datetime
        self.__folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        self._logger_path = os.path.join("logs",  self.__folder_name)
        self._logger_id = _logger.add(os.path.join(self._logger_path, "main.log"), format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <red>{level}</red>: {message}", delay=True)

    @property
    def logger_path(self):
        return self._logger_path

    def __new__(cls):  
        """This class uses singleton mode
        """
        if not hasattr(cls, '_instance'):
            orig = super(Logger, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance

    def set_is_use_logger(self, is_use_logger):
        self.__is_use_logger = is_use_logger
        if not is_use_logger:
            _logger.remove(self._logger_id)
            self._logger_id = None
            self._logger_path = None
        return self

    def set_folder_name(self, folder_name, remove_existing_folder = True):
        # if exists, remove it
        _logger.remove(self._logger_id)
        dirpath = Path('logs', folder_name)
        if dirpath.exists() and dirpath.is_dir():
            if remove_existing_folder:
                shutil.rmtree(dirpath)
                _logger.info("[+] The folder \"" + folder_name + "\" exits! Remove it successfully!")
            else:
                self.info("[+] Folder \"" + folder_name + "\" already exists! Logger file will be added!")
        self.__folder_name = folder_name
        self._logger_path = os.path.join("logs",  self.__folder_name)
        self._logger_id = _logger.add(os.path.join(self._logger_path, "main.log"), format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <red>{level}</red>: {message}", delay=True)
        return self

    def set_is_save_json(self, is_save_json):
        self.__is_save_json = is_save_json
        return self

    def save_to_json(self, **kwargs): # 
        """ if IS_SAVE_JSON = False, then save it to memmory
        """
        data_str = json.dumps(kwargs)
        if self.__is_save_json:
            if not self.__is_use_logger:
                raise Exception("Logger must be used to save json. Please call \"logger.set_is_use_logger(True)\".") 
            try:
                file_object = open(os.path.join("logs", self.__folder_name, "main.json"), 'a')
            except PermissionError:
                file_object = open(os.path.join("logs", self.__folder_name, "main.json"), 'a')
            file_object.write(data_str + "\n")
            file_object.close()
        else:
            self.memory_json = kwargs

    def read_from_json(self, folder_name = None, no_iter = -1):
        """ if IS_SAVE_JSON = False and folder_name = None, then read it from memmory
        """
        if self.__is_save_json == False and folder_name is None:
            return self.memory_json

        if folder_name is None:
            folder_name = self.__folder_name
        try:
            file_object = open(os.path.join("logs", folder_name, "main.json"))
            for i, line in enumerate(file_object):
                if i == no_iter:
                    return json.loads(line)
            if no_iter == -1:
                return json.loads(line)
            raise Exception("The number of iteration exceeds the maximum iteration number!")
        except FileNotFoundError:
            raise Exception("The json file is not saved in the log file \"" + folder_name + "\". Please set \"is_save_json = True\" in the \"learn\" method.")

    def set_level(self, level):
        """
        Set logging threshold on current logger.
        """
        self.__min_level = level

    def debug(self, *args):
        if self.__min_level <= self.DEBUG:
            _logger.debug(*args)

    def info(self, *args):
        if self.__min_level <= self.INFO:
            _logger.info(*args)

    def warn(self, *args):
        if self.__min_level <= self.WARN:
            _logger.warning(*args)

    def error(self, *args):
        if self.__min_level <= self.ERROR:
            _logger.error(*args)

# Global logger
logger = Logger() 