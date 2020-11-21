from loguru import logger as _logger
import os

def logger_init(**args):
    log_path = os.path.join("logs", args["file_name"], "_result.log")
    logger_id = _logger.add(log_path, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}")
    for arg in args:
        _logger.debug("[+] " + arg + ": " +  str(args[arg]))
    return logger_id
    
def logger_destroy(logger_id):
    _logger.remove(logger_id)

class Logger(object):
    def debug(self, *args):
        _logger.debug(args)
        
logger = Logger()