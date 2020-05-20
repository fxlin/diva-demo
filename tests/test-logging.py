#!/usr/bin/env

import sys, logging, coloredlogs

FORMAT = '%(levelname)8s {%(pathname)s:%(lineno)d} %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

coloredlogs.install(fmt=FORMAT, level='DEBUG', logger=logger)

if __name__ == '__main__':
    logger.debug("This is a debug log")
    logger.info("This is an info log")
    logger.critical("This is critical")
    logger.error("An error occurred")