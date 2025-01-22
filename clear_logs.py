import os

path_to_logs = os.path.join(os.getcwd(), 'logs')

log_files = os.listdir(path_to_logs)

for file in log_files:
    path_to_file = os.path.join(path_to_logs, file)
    try:
        os.remove(path_to_file)
        print(f"Removed file {file} ")
    except Exception as e:
        print(f"Failed to delete file", file)
        print(e)