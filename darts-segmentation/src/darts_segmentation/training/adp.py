"""Abstract Data Parallelism (ADP) module for DARTS Segmentation."""

import logging
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from typing import TypeVar

logger = logging.getLogger(__name__.replace("darts_", "darts."))

RunI = TypeVar("I")
RunO = TypeVar("O")


def _adp(
    process_inputs: list[RunI],
    is_parallel: bool,
    devices: list[int],
    available_devices: Queue,
    _run: Callable[[RunI], RunO],
) -> Generator[tuple[RunI, RunO], None, None]:
    # Handling different parallelization strategies
    if is_parallel:
        logger.debug("Using parallel strategy for ADP")
        for device in devices:
            logger.debug(f"Adding device {device} to available devices queue")
            available_devices.put(device)
        with ProcessPoolExecutor(max_workers=len(devices)) as executor:
            futures = {executor.submit(_run, inp): inp for inp in process_inputs}

            for future in as_completed(futures):
                inp = futures[future]
                try:
                    output = future.result()
                except (KeyboardInterrupt, SystemError, SystemExit):
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    logger.error(f"Error in {inp}: {e}", exc_info=True)
                    continue

                yield inp, output
    else:
        logger.debug("Using serial strategy for ADP")
        for inp in process_inputs:
            try:
                output = _run(inp)
            except Exception as e:
                logger.error(f"Error in {inp}: {e}", exc_info=True)
                continue
            yield inp, output
