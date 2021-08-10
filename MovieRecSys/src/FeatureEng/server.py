import socket
import time
import select

def do_send(sock, msg, timeout):
    readers = []
    writers = [sock]
    excepts = [sock]
    rxs, txs, exs = select.select(readers, writers, excepts, timeout)
    if sock in exs:
        return False
    elif sock in txs:
        sock.send(msg)
        return True
    else:
        return False

host = 'localhost'
port = 9999

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host, port))
s.listen(1)

print("waiting for client")
conn, addr = s.accept()
print("client connected")
conn.setblocking(0)

batchSize = 50
idle = 1.
count = 0
running = 1
while running:
    try:
        sc = 0
        while (sc < batchSize):
            if do_send (conn, 'Hello World ' + repr(count) + '\n', 1.0):
                sc += 1
                count += 1
        print("sent " + repr(batchSize) + ", waiting " + repr(idle) + " seconds")
        time.sleep(idle)
    except socket.error:
        conn.close()
        running = 0

print("done")
