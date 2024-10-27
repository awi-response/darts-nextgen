"""Earth Engine utilities."""

import logging

import ee

# geemap is not used yet
# import geemap

logger = logging.getLogger(__name__)


def init_ee(project: str):
    """Initialize Earth Engine. Authenticate if necessary.

    Args:
        project (str): The project name.

    """
    logger.debug("Initializing Earth Engine")
    try:
        ee.Initialize(project=project)
        # geemap.ee_initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        logger.debug("Initializing Earth Engine failed, trying to authenticate before")
        ee.Authenticate()
        ee.Initialize(project=project)
        # geemap.ee_initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    logger.debug("Earth Engine initialized")
