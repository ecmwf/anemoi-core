# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import sys
import time


def get_code_logger(name: str, debug: bool = True) -> logging.Logger:
    """Returns a logger with a custom level and format.

    We use ISO8601 timestamps and UTC times.

    Parameters
    ----------
    name : str
        Name of logger object
    debug : bool, optional
        set logging level to logging.DEBUG; else set to logging.INFO, by default True

    Returns
    -------
    logging.Logger
        Logger object
    """
    # create logger object
    logger = logging.getLogger(name=name)
    if not logger.hasHandlers():
        # logging level
        level = logging.DEBUG if debug else logging.INFO
        # logging format
        datefmt = "%Y-%m-%dT%H:%M:%SZ"
        msgfmt = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName).30s] [%(levelname)s] %(message)s"
        # handler object
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(msgfmt, datefmt=datefmt)
        # record UTC time
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)

    return logger
