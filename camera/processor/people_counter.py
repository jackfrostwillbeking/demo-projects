from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np
import cv2


CAMERA_WIDTH = 800    # default = 320 PiCamera image width can be greater if quad core RPI
CAMERA_HEIGHT = 608   # default = 240 PiCamera image height
CAMERA_HFLIP = False  # True=flip camera image horizontally
CAMERA_VFLIP = False  # True=flip camera image vertically
CAMERA_ROTATION = 0   # Rotate camera image valid values 0, 90, 180, 270
CAMERA_FRAMERATE = 20 # default = 25 lower for USB Web Cam. Try different settings
WINDOW_BIGGER = 2   # Resize multiplier for Movement Status Window
                    # if gui_window_on=True then makes opencv window bigger
                    # Note if the window is larger than 1 then a reduced frame rate will occur
MIN_AREA = 800 

x_center = CAMERA_WIDTH/2
y_center = CAMERA_HEIGHT/2
x_max = CAMERA_HEIGHT
y_max = CAMERA_WIDTH
x_buf = CAMERA_WIDTH/10
y_buf = CAMERA_HEIGHT/10

big_w = int(CAMERA_WIDTH * WINDOW_BIGGER)
big_h = int(CAMERA_HEIGHT * WINDOW_BIGGER)
cx, cy, cw, ch = 0, 0, 0, 0

movelist_timeout = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX


class PeopleCounter(object):
    def __init__(self, flip=True):
        self.vs = PiVideoStream(resolution=(800, 608)).start()
        self.flip = flip
        time.sleep(2.0)
        self.firstFrame = None
        self.move_time = time.time()
        self.movelist = []
        self.enter = 0
        self.leave = 0

    def __del__(self):
        self.vs.stop()


    def crossed_y_centerline(self, enter, leave, movelist):
        # Check if over center line then count
        if len(movelist) > 1:  # Are there two entries
            if ( movelist[0] <= y_center
                   and  movelist[-1] > y_center + y_buf ):
                leave += 1
                movelist = []
            elif ( movelist[0] > y_center
                   and  movelist[-1] < y_center - y_buf ):
                enter += 1
                movelist = []
        return enter, leave, movelist


    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        frame = self.process_image(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def process_image(self, frame):
        motion_found = False
        biggest_area = MIN_AREA
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.firstFrame is None:
            self.firstFrame = gray
            return frame
          
        cv2.line( frame,( 0, y_center ),( x_max, y_center ),(255, 0, 0), 2 )

        frameDelta = cv2.absdiff(self.firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            for c in cnts:
                if cv2.contourArea(c) < 700:
                    continue
                found_area = cv2.contourArea(c)
                if found_area > biggest_area:
                    motion_found = True
                    biggest_area = found_area
                    (x, y, w, h) = cv2.boundingRect(c)
                    cx = int(x + w/2)
                    cy = int(y + h/2)
                    cw, ch = w, h

            if motion_found:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 2)
                move_timer = time.time() - self.move_tim
                if (move_timer >= movelist_timeout):
                    self.movelist = []
                self.move_time = time.time()

                old_enter = self.enter
                old_leave = self.leave
                self.movelist.append(cy)
                self.enter, self.leave, self.movelist = self.crossed_y_centerline(self.enter, self.leave, self.movelist)

                if not self.movelist:
                    if self.enter > old_enter:
                        prefix = 'enter'
                    elif self.leave > old_leave:
                        prefix = 'leave'
                    else:
                        prefix = 'error'

        img_text = ("LEAVE %i : ENTER %i" % (self.enter, self.leave))
        cv2.putText(frame, img_text, (45, 25), font, 1.0, (0, 0, 255), 2)
        
        return frame
