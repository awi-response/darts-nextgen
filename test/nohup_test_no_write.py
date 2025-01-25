import time

for i in range(0, 10):
    print(f"We are doing round {i} and will rest for one minute.")
    time.sleep(60)
print(f"We finished the whole thing")