import zmq, sys

file = sys.argv[-1]

if file == 'inference_client.py':
    print ("Error loading video")
    exit()

context = zmq.Context()

#  Socket to talk to server
#print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
#print("connected")

#print("Sending file name: " + file)
socket.send_string(sys.argv[1])
socket.socket.recv_string()
socket.send_string(sys.argv[2])
socket.socket.recv_string()
socket.send_string(sys.argv[3])
socket.socket.recv_string()
socket.send_string(sys.argv[4])
socket.socket.recv_string()
socket.send_string(file)
#print("File name sent, waiting for respond ... ")
print (socket.recv_string())
#print("Done")