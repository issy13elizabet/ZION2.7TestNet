import subprocess
import time
import threading

cmd = ['miners/SRBMiner-Multi-latest/SRBMiner-Multi-2-9-8/SRBMiner-MULTI.exe', '--algorithm', 'kawpow', '--pool', 'pool.ravenminer.com:3838', '--wallet', 'test_wallet', '--password', 'x', '--gpu-boost', '3', '--disable-cpu', '--api-enable', '--api-port', '5380', '--gpu-platform', '1', '--gpu-intensity', '22', '--gpu-worksize', '16', '--gpu-threads', '1', '--gpu-memclock', 'boost', '--gpu-coreclock', '+50', '--gpu-fan', '70-85']
print('Starting SRBMiner...')
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd='miners/SRBMiner-Multi-latest/SRBMiner-Multi-2-9-8')

def read_output():
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f'SRBMiner: {line.strip()}')

thread = threading.Thread(target=read_output, daemon=True)
thread.start()

time.sleep(10)
print('Terminating SRBMiner...')
proc.terminate()
try:
    proc.wait(timeout=2)
except:
    proc.kill()