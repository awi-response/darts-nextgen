"""Earth Engine utilities."""

import logging

import ee

# geemap is not used yet
# import geemap

logger = logging.getLogger(__name__)


def init_ee(project: str | None = None, use_highvolume: bool = True) -> None:
    """Initialize Earth Engine. Authenticate if necessary.

    Args:
        project (str): The project name.
        use_highvolume (bool): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).

    """
    logger.debug(f"Initializing Earth Engine with project {project} {'with high volume' if use_highvolume else ''}")
    opt_url = "https://earthengine-highvolume.googleapis.com" if use_highvolume else None
    try:
        ee.Initialize(project=project, opt_url=opt_url)
        # geemap.ee_initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        logger.debug("Initializing Earth Engine failed, trying to authenticate before")
        ee.Authenticate()
        ee.Initialize(project=project, opt_url=opt_url)
        # geemap.ee_initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    logger.debug("Earth Engine initialized")
