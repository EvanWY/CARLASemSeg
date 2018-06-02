import zmq, sys

file = sys.argv[-1]

if file == 'run.py':
    print ("Error loading video")
    quit

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
print("connected")

print("Sending file name: " + file)
socket.send(file)
print("File name sent, waiting for respond ... ")
message = socket.recv()
print("Done")

print (message)