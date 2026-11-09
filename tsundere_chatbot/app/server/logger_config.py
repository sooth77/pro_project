"""로깅 설정. 여러 번 불러도 핸들러가 겹치지 않음."""
import logging
import sys


def setup_logger(name="ml_api", level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger
