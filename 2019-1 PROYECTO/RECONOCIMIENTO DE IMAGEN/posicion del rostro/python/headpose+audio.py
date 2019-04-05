#!/usr/bin/env python
# 2018-05-20 I aim to have this run in a while loop getting the camera frame

from imutils.video import VideoStream
from imutils import face_utils
from pyAudioAnalysis import audioTrainTest as aT
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import time
import pyaudio
import wave
import subprocess
import os
import time
import threading
import audioop

class Recorder():
    #Defines sound properties like frequency and channels
    def __init__(self, chunk=1024, channels=2, rate=44100):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.vol = 0

    #Start recording sound
    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        #Set running to True and reset previously recorded frames
        self._running = True
        self._frames = []
        #Create pyaudio instance
        p = pyaudio.PyAudio()
        #Open stream
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        # To stop the streaming, new thread has to set self._running to false
        # append frames array while recording
        while(self._running):
            data = stream.read(self.CHUNK)
            self._frames.append(data)
            #dat=np.frombuffer(data,dtype=np.int16)
            #self.vol=np.average(np.abs(dat))*2
            rms=audioop.rms(data,2)
            if(rms>0):
                self.vol = 20 * np.log10(rms)

        # Interrupted, stop stream and close it. Terinate pyaudio process.
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Sets boolean to false. New thread needed.
    def stop(self):
        self._running = False

    def getVol(self):
        #bars="#"*int(50*self.vol/2**16)
        # bars = "|"*int(100/(self.vol+1))
        print ("|" * int((self.vol-35)*2))
        #print (self.vol)
        #print(bars)

  
        
    #Save file to filename location as a wavefront file.
    def save(self, filename):
        #print("Saving")
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        #print("Saved")

    # Delete a file
    @staticmethod
    def delete(filename):
        os.remove(filename)

    # Convert wav to mp3 with same name using ffmpeg.exe
    @staticmethod
    def wavTomp3(wav):
        mp3 = wav[:-3] + "mp3"
        # Remove file if existent
        if os.path.isfile(mp3):
            Recorder.delete(mp3)
        # Call CMD command
        subprocess.call('ffmpeg -i "'+wav+'" "'+mp3+'"')


# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#        help="path to facial landmark predictor")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] CARGANDO INFORMACION DE DETECCION")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] ACTIVANDO CAMARA...")

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# 400x225 to 1024x576
frame_width = 1024
frame_height = 576

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# loop over the frames from the video stream
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (359, 391),     # Nose tip 34
                            (399, 561),     # Chin 9
                            (337, 297),     # Left eye left corner 37
                            (513, 301),     # Right eye right corne 46
                            (345, 465),     # Left Mouth corner 49
                            (453, 469)      # Right mouth corner 55
                        ], dtype="double")

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])
rec = Recorder()
print("INICIA GRABACION DE AUDIO")
rec.start()
estabamirando=[0,0,0,0,0]
start=[0,0,0,0,0]
end = [0,0,0,0,0]
numeroCara=0
segtotal=0
maxcaras=0
bigstart= time.time() #timestamp de cuando inicia el procesamiento
bigend=0
while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=1024, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        if len(rects) > 0:
                text = "{} CARAS DETECTADAS".format(len(rects))
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)

        ##
        if(maxcaras<len(rects)):
                maxcaras = len(rects)
        caras=0
        carasmirando=0
        ##
       
        # loop over the face detections
        numeroCara=0
        for rect in rects:
                        
                # compute the bounding box of the face and draw it on the
                # frame
                        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 0, 255), 1)
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw each of them
                        for (i, (x, y)) in enumerate(shape):
                                if i == 33:
                                        #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        image_points[0] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                elif i == 8:
                    #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        image_points[1] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                elif i == 36:
                                        #something to our key landmarks
                                        # save to our new key point list
                                        # i.e. keypoints = [(i,(x,y))]
                                        image_points[2] = np.array([x,y],dtype='double')
                                        # write on frame in Green
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                elif i == 45:
                                        #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        image_points[3] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                elif i == 48:
                    #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        image_points[4] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                elif i == 54:
                    #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        image_points[5] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                else:
                    #everything to all other landmarks
                    # write on frame in Red
                                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                        focal_length = size[1]
                        center = (size[1]/2, size[0]/2)
                        camera_matrix = np.array([[focal_length,0,center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")

                        #print "Camera Matrix :\n {0}".format(camera_matrix)

                        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

                        
                        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)#flags=cv2.CV_ITERATIVE)

                        #text = "{} rotacion".format(rotation_vector)
                        #
                        #cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                        #text = "{} traslacion".format(translation_vector)
                        #cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
                        proj_matrix = np.hstack((rvec_matrix, translation_vector))

                        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                        caras+=1

                        #print("ROLL:"+str(euler_angles[2])+" - PITCH:"+str(euler_angles[1])+" - YAW:"+str(euler_angles[0]))
                        #YAW ES EN ANGULO DE LA CABEZA ARRIBA - ABAJO
                        if euler_angles[0]<-150 and euler_angles[0]>-170:
                                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
                                #print("ESTAS MIRANDO "+str(numeroCara)+" : "+str(euler_angles[0]));
                                carasmirando+=1
                                if estabamirando[numeroCara] == 0:
                                       estabamirando[numeroCara] = 1
                                       #segtotal+=1
                                       start[numeroCara]=time.time()
                                        
                        else:
                                #print("NO ESTAS MIRANDO "+str(numeroCara)+" : "+str(euler_angles[0]));
                                if estabamirando[numeroCara] == 1:
                                        estabamirando[numeroCara] = 0
                                        end[numeroCara] = time.time()
                                        segtotal+=(end[numeroCara]-start[numeroCara])
                                        end[numeroCara] = 0
                                        start[numeroCara] = 0
                                                
                        #print "Rotation Vector:\n {0}".format(rotation_vector)
                        #print "Translation Vector:\n {0}".format(translation_vector)
                        # Project a 3D point (0, 0 , 1000.0) onto the image plane
                        # We use this to draw a line sticking out of the nose_end_point2D
                        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                        for p in image_points:
                                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

                        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                        cv2.line(frame, p1, p2, (255,0,0), 2)
                        numeroCara+=1
        
       # print("CARAS ENCONTRADAS:("+str(caras)+") - CARAS MIRANDO:("+str(carasmirando)+")")

        text = "{} CARAS MIRANDO".format(carasmirando)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        rec.getVol()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                bigend=time.time()              
                print("TOTAL CARAS ENCONTRADAS: "+str(maxcaras))
                print("TIEMPO MIRANDO TOTAL: "+str('{0:.3g}'.format(segtotal/60))+" minutos")
                print("TIEMPO TOTAL: "+str('{0:.3g}'.format((bigend-bigstart)/60))+" minutos")
                print("PORCENTAJE DE ATENCION: "+str('{0:.3g}'.format(((segtotal)/((bigend-bigstart)*maxcaras))*100))+"%")
                #print("Stop recording")
                rec.stop()
                print("GUARDANDO ARCHIVO DE AUDIO")
                rec.save("test.wav")
                time.sleep(3)
                print("RESULTADO DEL ANALISIS DE AUDIO:")
                print(aT.fileClassification("test.wav", "model/GBSpeechv2","svm"))

                break

#print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
