from dask.distributed import Client

from dask.distributed import LocalCluster
cluster = LocalCluster()          # Fully-featured local Dask cluster
client = cluster.get_client()

# client = Client()  # Connect to an existing cluster or create a new one

# Get a list of worker addresses
worker_addresses = list(client.scheduler_info()['workers'].keys())

client.retire_workers(workers=worker_addresses)

print(worker_addresses)