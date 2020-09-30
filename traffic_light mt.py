import cv2
import numpy as np
import time 
import datetime
import threading
from queue import Queue


SHUTDOWN_SIG = False

def nothing(x): pass 

class Camera: 
    
    def __init__(self, path, RECORD=False,HILENS=False):
        self.HILENS = HILENS
        self.RECORD = RECORD
        if HILENS: import hilens
        self.disp = ""
        if path != "":
            self.cap = cv2.VideoCapture(path) 
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 520) 
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 
        else:
            self.cap = hilens.VideoCapture()
            self.disp = hilens.Display(hilens.HDMI)

        if RECORD:
            fps = 20 
            size = (520, 360) 
            format = cv2.VideoWriter_fourcc('M','J','P','G') 
            self.src_video_writer = cv2.VideoWriter("src_output.avi", format, fps, size) 

 
    def getFrame(self):
        frame = None
        if self.HILENS:
            frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
        else:
            ret, frame = self.cap.read() 
            if not ret: 
                frame = None
        
        ret = frame

        if frame is not None:
            if self.RECORD:
                self.src_video_writer.write(frame)
            # if self.HILENS == False:
            #     cv2.imshow("src", frame) 
    
        return ret


    def display(self, frame=None):
        if self.HILENS == False:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # dim = (720, 520)
            # # resize image
            # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            # self.disp.show(resized)
            # self.dst_video_writer.write(frame)
            cv2.imshow("results", frame) 
        else:
            # resize image
            frame = cv2.resize(frame, (1280, 720))
            output_yuv = hilens.cvt_color(frame, hilens.RGB2YUV_NV21)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # self.dst_video_writer.write(resized)
            self.disp.show(output_yuv)

    def __del__(self): 
        pass 

    def close(self): 
        self.__del__() 
  
  
class PictureProcessor():
    def __init__(self, HILENS = True, RECORD = False, TrackBar=False):
        self.__fps = 0
        self.__fps_start = time.time()

        if RECORD:
            fps = 20 
            size = (520, 360) 
            format_ = cv2.VideoWriter_fourcc('M','J','P','G') 
            self.dst_video_writer = cv2.VideoWriter("dst_output.avi", format_, fps, size) 

        if HILENS: self.disp = hilens.Display(hilens.HDMI) 

        self.threshold = 0.78

        self.low_red_hsv  = (170,120,120)
        self.high_red_hsv = (180,255,255)
        self.low_yellow_hsv  = (15,110,110)
        self.high_yellow_hsv = (30,255,255)
        self.low_green_hsv  = (80,125,130)
        self.high_green_hsv = (90,255,255)

        w = 520 
        h = 360 
        self.roi_rect = [(int(w*1/4), int(h*2/5)), (int(w*3/4), int(h*2/3))] 

        if TrackBar:
            cv2.namedWindow("param", cv2.WINDOW_NORMAL) 
            cv2.createTrackbar("l_h_r", "param", self.low_red_hsv[0], 180, nothing) 
            cv2.createTrackbar("l_s_r", "param", self.low_red_hsv[1], 255, nothing) 
            cv2.createTrackbar("l_v_r", "param", self.low_red_hsv[2], 255, nothing) 
            cv2.createTrackbar("h_h_r", "param", self.high_red_hsv[0], 180, nothing) 
            cv2.createTrackbar("h_s_r", "param", self.high_red_hsv[1], 255, nothing) 
            cv2.createTrackbar("h_v_r", "param", self.high_red_hsv[2], 255, nothing) 
            cv2.createTrackbar("l_h_y", "param", self.low_yellow_hsv[0], 180, nothing) 
            cv2.createTrackbar("l_s_y", "param", self.low_yellow_hsv[1], 255, nothing) 
            cv2.createTrackbar("l_v_y", "param", self.low_yellow_hsv[2], 255, nothing) 
            cv2.createTrackbar("h_h_y", "param", self.high_yellow_hsv[0], 180, nothing) 
            cv2.createTrackbar("h_s_y", "param", self.high_yellow_hsv[1], 255, nothing) 
            cv2.createTrackbar("h_v_y", "param", self.high_yellow_hsv[2], 255, nothing) 
            cv2.createTrackbar("l_h_g", "param", self.low_green_hsv[0], 180, nothing) 
            cv2.createTrackbar("l_s_g", "param", self.low_green_hsv[1], 255, nothing) 
            cv2.createTrackbar("l_v_g", "param", self.low_green_hsv[2], 255, nothing) 
            cv2.createTrackbar("h_h_g", "param", self.high_green_hsv[0], 255, nothing) 
            cv2.createTrackbar("h_s_g", "param", self.high_green_hsv[1], 255, nothing) 
            cv2.createTrackbar("h_v_g", "param", self.high_green_hsv[2], 255, nothing) 

        self.TrackBar = TrackBar
        self.HILENS = HILENS
        self.RECORD = RECORD

        template_paths = ["Images/red_template_1.jpg", "Images/yellow_template_1.jpg"]
        self.templates = list()  
        for path in template_paths: 
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # self.templates[-1] = cv2.convertScaleAbs(self.templates[-1], alpha=1.5, beta=-100)
            self.templates.append(cv2.resize(template, (9,9))) 
            self.templates.append(cv2.resize(template, (15,15)))
            self.templates.append(cv2.resize(template, (21,21)))
            self.templates.append(cv2.resize(template, (27,27))) 
            self.templates.append(cv2.resize(template, (33,33)))

        for t in self.templates: print(t.shape)

        self.labels = ["red_stop", "green_go", "yellow_back"] 
        self.colors = [(0,0,255), (0,255,0), (0,255,255)]

    def _roi(self, frame):
        cv2.rectangle(frame, self.roi_rect[0], self.roi_rect[1], (255,255,255), 1)
        return frame[self.roi_rect[0][1]:self.roi_rect[1][1], self.roi_rect[0][0]:self.roi_rect[1][0]] 


    def __calFPS(self):
        if self.__fps == 30:
            duration = time.time() - self.__fps_start
            print("fps:", self.__fps / duration)
            self.__fps = 0
            self.__fps_start = time.time()
        else:
            self.__fps = self.__fps + 1

    def display(self, frame):
        if self.HILENS == False:
            cv2.imshow("results", frame) 
        else:
            # resize image
            frame = cv2.resize(frame, (1280, 720))
            output_yuv = hilens.cvt_color(frame, hilens.RGB2YUV_NV21)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # self.dst_video_writer.write(resized)
            self.disp.show(output_yuv)

    def detect(self, img): 
        global SHUTDOWN_SIG
        self.__calFPS()
        img = cv2.resize(img, (520, 360))
        self.__detect_img = self._roi(img)

        img_gray = cv2.cvtColor(self.__detect_img, cv2.COLOR_BGR2GRAY) 
        for template in self.templates: 
            try: 
                w, h = template.shape[::-1]
                w += 4; h += 4 
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where( res >= self.threshold)

                for pt in zip(*loc[::-1]): 
                    pt = (pt[0] - 2, pt[1] - 2)  
                    hsv = cv2.cvtColor(self.__detect_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w], cv2.COLOR_BGR2HSV) 
                    mask_red = cv2.inRange(hsv, self.low_red_hsv, self.high_red_hsv) 
                    mask_green = cv2.inRange(hsv, self.low_green_hsv, self.high_green_hsv)
                    mask_yellow = cv2.inRange(hsv, self.low_yellow_hsv, self.high_yellow_hsv)
                    sums = [ np.sum(mask) for mask in [mask_red, mask_green, mask_yellow] ]
                    max_loc = np.argmax(sums) 
                    if sums[max_loc] < w*h * 255 * 0.1: 
                        cv2.rectangle(self.__detect_img, pt, (pt[0] + w, pt[1] + h), (125,125,0), 2)
                        continue
                    cv2.rectangle(self.__detect_img, pt, (pt[0] + w, pt[1] + h), self.colors[max_loc], 2)
                    results = { "bndbox": (pt[0],pt[1],w,h), "label": self.labels[max_loc] } 

                    # Update roi. 
                    tole = 50 
                    pp1, pp2 = self.roi_rect 
                    max_x = 520
                    max_y = 360
                    self.roi_rect = [( max(pt[0]-tole+pp1[0], 0), max(pt[1]-tole+pp1[1], 0) ), 
                                     ( min(pt[0]+w+tole+pp1[0], max_x), min(pt[1]+h+tole+pp1[1], max_y) ) ] 

                    # @return: dict: {"bndbox", (x,y,w,h), "label", "red"|"green"|"yellow"} 
                    # print(results)
                    self.__detect_img = cv2.putText(self.__detect_img, results["label"], (pt[0],pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[max_loc], 2) 
                    if self.HILENS == False:
                        cv2.imshow("result", img)
                        if cv2.waitKey(1) == 27: 
                            SHUTDOWN_SIG = True
                    return (results["label"], self.__detect_img)
            except: print("roi error")

        # Reset roi. 
        w = 520 
        h = 360
        self.roi_rect = [(int(w*1/7), int(h*1/5)), (int(w*6/7), int(h*3/5))]
        if self.HILENS == False:
            cv2.imshow("result", img)
            if cv2.waitKey(1) == 27: 
                SHUTDOWN_SIG = True

        # @return: None
        return ("", self.__detect_img)


class CameraThreadGuard(threading.Thread):
    def __init__(self, video_path="", RECORD=False, HILENS=True):
        threading.Thread.__init__(self)
        self.__cam = Camera(video_path, RECORD, HILENS)
    
    def run(self):
        global src_img_buff, SHUTDOWN_SIG 
        try:
            while not SHUTDOWN_SIG:
                frame = self.__cam.getFrame()
                if frame is not None:
                    src_img_buff.put(frame)
                else: self.__cam.close(); break  
        finally:
            SHUTDOWN_SIG = True 
            print('cam ended')

class PictureProcessorThreadGuard(threading.Thread):
    def __init__(self, HILENS=True, RECORD=False, TrackBar=False):
        threading.Thread.__init__(self)
        self.__pp = PictureProcessor(HILENS, RECORD, TrackBar)
    
    def run(self):
        global src_img_buff
        try: 
            while not SHUTDOWN_SIG:
                self.__pp.detect(src_img_buff.get())
        finally: 
            print('pp ended') 


if __name__ == "__main__":
    src_img_buff = Queue(10)
    # video_path = "src_output2.mp4"
    video_path = "C:/Users/Acer/AppData/Local/Packages/903DB504.QQ_a99ra4d2cbcxa/LocalState/User/3437859177/NetworkFile/video0836 (9).mp4"
    # video_path = "C:\\Users\\CHKB\\Desktop\\src_output2.avi"
    # video_path = "/dev/video10"
    # cam = Camera(path="/dev/video10")
    cam = Camera(path=video_path, RECORD=False, HILENS=False)
    picture_processor = PictureProcessor(False)

    cam_thread = CameraThreadGuard(video_path=video_path, RECORD=False, HILENS=False)
    pp_thread = PictureProcessorThreadGuard(False)

    thread_lists = [cam_thread, pp_thread]

    for t in thread_lists:
        t.setDaemon(True)
    print("start")
    try:
        for t in thread_lists:
            t.start()

        # Wait for complete. 

        for t in thread_lists:
            t.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        cv2.destroyAllWindows() 
        print('exit')