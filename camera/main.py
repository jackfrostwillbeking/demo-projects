from flask import Flask, render_template, Response
# from processor.simple_streamer import Processor
# from processor.pedestrian_detector import Processor
# from processor.motion_detector import Processor
from processor.people_counter import PeopleCounter as Processor
# from processor.person_detector import Processor
import time
import threading

video_camera = Processor(flip=False)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
