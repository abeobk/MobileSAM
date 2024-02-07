"""
    TCP server to inference SAM remotely
"""

print("Starting SAM server..." );

import sys
import time
import json
import math
import itertools
import socket
import numpy as np
import cv2

HOST=""
PORT=8727
#PORT = sys.argv[1];


#HELPER
class TicToc:
    t0 = 0;
    #start timer
    def tic(self, msg = None):
        self.t0 = time.monotonic();
        if(msg is not None):
            print(msg + "...", end="")

    #stop timer
    def toc(self):
        dt = time.monotonic() - self.t0;
        print("("+str(round(dt,2))+" s)");
        

tt = TicToc();

#LOADING LIBRARIES
tt.tic("Loading libraries");
import torch
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
tt.toc();

#model path
sam_checkpoint = "weights/mobile_sam.pt"
model_type = "vit_t"
device = "cuda" if torch.cuda.is_available() else "cpu"

#LOAD MODELS
tt.tic("Loading SAM");
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval();
predictor = SamPredictor(sam)
tt.toc();

def send_ok(con:socket):
    con.sendall("OK\n".encode())

def send_err(con:socket, msg: str):
    con.sendall(f"ERROR: {msg}\n".encode())


#SERVER
print(f"Starting server...")
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))
    while True:
        try:
            print(f"Listerning on port {PORT}...")
            s.listen()
            con,client_addr = s.accept()
            with con:
                print(f"Connected by {client_addr}")
                connected=True
                while connected:
                    try:
                        jstr="";
                        #get a line
                        while connected:
                            data = con.recv(1)
                            c = chr(data[0])
                            #found new line, end of cmd
                            if(c=='\r' and chr(con.recv(1)[0])=='\n'):
                                break;
                            jstr+=c

                        if jstr:
                            print(">"+jstr)
                            cmd = json.loads(jstr)
                            # set a new image
                            if(cmd["name"]=="set_img"):
                                imgsz = cmd["size"]
                                buf = bytearray(imgsz)
                                # send OK, start receiving
                                send_ok(con);
                                byte_reads = con.recv_into(buf,imgsz)
                                if(byte_reads == imgsz):
                                    #decode image
                                    img = cv2.imdecode(np.frombuffer(buf,dtype=np.uint8), cv2.IMREAD_COLOR)
                                    # set image
                                    tt.tic(f"Inferencing ({device})");
                                    predictor.set_image(img);
                                    tt.toc();
                                    #send OK
                                    send_ok(con);
                                else:
                                   raise Exception(f"Corrupted data when processing {jstr}"); 

                            # predict on current image
                            elif(cmd["name"]=="predict"):
                                boxes =  np.array(cmd["box"]) if "box" in cmd else None
                                points = np.array(cmd["point"]) if "point" in cmd else None
                                labels = np.array(cmd["label"]) if "label" in cmd else None
                                multimask = cmd["multimask"] if "multimask" in cmd else False
                                #send OK
                                send_ok(con);
                                masks, scores, logits = predictor.predict(
                                    point_coords=points,
                                    point_labels=labels,
                                    box = boxes,
                                    multimask_output=multimask,
                                )
                                id = scores.tolist().index(max(scores))
                                mask = masks[id].astype(np.uint8)*255;
                                maskbuf = cv2.imencode(".png",mask)[1].tobytes()
                                con.sendall(f'{{"mask":{id},"score":{scores[id]},"size":{len(maskbuf)}}}\n'.encode())
                                con.sendall(maskbuf)
                                send_ok(con);
                            else:
                                print(f"Unknown command {jstr}")
                    except Exception as ex:
                        print("ERROR: " + str(ex))
                        send_err(con,str(ex))
                    time.sleep(0.010); #100 hz polling
        except Exception as ex: 
            print("Opps! "+str(ex))


