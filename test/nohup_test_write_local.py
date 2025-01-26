import time
import os

path_to_file = '/home/toddn/darts-nextgen/test/nohup_test.txt'

for i in range(0, 10):
    print(f"We are doing round {i} and will rest for one minute.")
    if os.path.exists(path_to_file):
        print(f"The file already exists")
        with open(path_to_file, 'a+') as f:
            current_line = "We did this for minute " + str(i) + '\n'
            f.write(current_line)
    else:
        print(f"The file does not already exist")
        with open(path_to_file, 'w+') as f:
            current_line = "We did this for minute " + str(i) + '\n'
            f.write(current_line)
    time.sleep(60)
print(f"We finished the whole thing")