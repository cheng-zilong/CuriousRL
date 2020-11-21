from loguru import logger
import os


def loguru_start(**args):
    log_path = os.path.join("logs", args["file_name"], "_result.log")
    logger_id = logger.add(log_path, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}")
    for arg in args:
        logger.debug("[+] " + arg + ": " +  str(args[arg]))
    return logger_id
    
def loguru_end(logger_id):
    logger.remove(logger_id)