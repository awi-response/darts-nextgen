"""Abstract Data Parallelism (ADP) module for DARTS Segmentation."""

import logging
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from typing import TypeVar

from darts_segmentation.training.train import DeviceConfig

logger = logging.getLogger(__name__.replace("darts_", "darts."))

RunI = TypeVar("I")
RunO = TypeVar("O")


def _adp(
    process_inputs: list[RunI],
    device_config: DeviceConfig,
    available_devices: Queue,
    _run: Callable[[RunI], RunO],
) -> Generator[tuple[RunI, RunO], None, None]:
    # Handling different parallelization strategies
    if device_config.strategy == "tune-parallel":
        for device in device_config.devices:
            available_devices.put(device)
        with ProcessPoolExecutor(max_workers=len(device_config.devices)) as executor:
            futures = {executor.submit(_run, inp): inp for inp in process_inputs}

            for future in as_completed(futures):
                inp = futures[future]
                try:
                    output = future.result()
                except Exception as e:
                    logger.error(f"Error in {inp}: {e}", exc_info=True)
                    continue

                yield inp, output
    else:
        for inp in process_inputs:
            try:
                output = _run(inp)
            except Exception as e:
                logger.error(f"Error in {inp}: {e}", exc_info=True)
                continue
            yield inp, output
