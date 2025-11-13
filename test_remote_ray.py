# To run this test remotely using Ray Job Submission, you can use the following command:
# uv run ray job submit \
#   --address http://127.0.0.1:8265 \
#   --working-dir . \
#   -- \
#   python test_remote_ray.py


import ray, os

ray.init()

@ray.remote
def check():
    return {
        "host": os.uname().nodename,
        "exists": os.path.exists("/data/workflows_optimization/rts_ray_pipeline"),
        "children": os.listdir("/data/workflows_optimization/rts_ray_pipeline")
    }

print(ray.get(check.remote()))
