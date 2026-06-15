import os
import subprocess
import time

# Start the FL server
server_process = subprocess.Popen(["python", "server.py"])
time.sleep(5)  # Wait for the server to start

client_processes = []

for aid in ["1", "2","3","4"]:
    env = os.environ.copy()
    env["bank_ID"] = aid
    p = subprocess.Popen(["python", "client.py"], env=env)
    client_processes.append(p)

# Wait for all clients to finish
for process in client_processes:
    process.wait()

# Terminate server after training
server_process.terminate()
server_process.wait()
print("\n🌍 Federated Learning Completed!")
