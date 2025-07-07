"""Predefined pipelines for DARTS."""

# from darts.pipelines.ray_v2 import AOISentinel2Pipeline as AOISentinel2RayPipeline
# from darts.pipelines.ray_v2 import PlanetPipeline as PlanetRayPipeline
# from darts.pipelines.ray_v2 import Sentinel2Pipeline as Sentinel2RayPipeline
from darts.pipelines.sequential_v2 import AOISentinel2Pipeline as AOISentinel2Pipeline
from darts.pipelines.sequential_v2 import PlanetPipeline as PlanetPipeline
from darts.pipelines.sequential_v2 import Sentinel2Pipeline as Sentinel2Pipeline
