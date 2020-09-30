import cv2
import numpy as np
import time 
import threading
from queue import Queue
from utils import *
from socket import *


src_img_buff = Queue(1)
results_buff = Queue(1)


SHUTDOWN_SIG = False

def nothing(x): pass 

class Camera: 
    
    def __init__(self, path, RECORD=False,HILENS=False):
        self.HILENS = HILENS
        self.RECORD = RECORD
        if HILENS: import hilens
        self.disp = ""
        if HILENS:
            import hilens
            self.cap = hilens.VideoCapture()
            
        else:
            self.cap = cv2.VideoCapture(path) 
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450) 
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320) 

        if RECORD:
            fps = 20 
            size = (450, 320) 
            format = cv2.VideoWriter_fourcc('M','J','P','G') 
            self.src_video_writer = cv2.VideoWriter("src_output.avi", format, fps, size) 

    def adjust_gamma(self, image, gamma=0.5):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table) 
 
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
            ret = self.adjust_gamma(ret, 0.7)
            # ret = cv2.erode(ret, np.ones((3,3), dtype=np.uint8)) 
            # if self.HILENS == False:
            #     cv2.imshow("src", frame) 
        return ret


    def display(self, frame=None):
        if self.HILENS == False:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # dim = (720, 480)
            # # resize image
            # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            # self.disp.show(resized)
            # self.dst_video_writer.write(frame)
            cv2.imshow("results", frame) 
        else:
            import hilens
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

        if HILENS: 
            import hilens
            self.disp = hilens.Display(hilens.HDMI) 

        self.threshold = 0.83

        self.low_red_hsv  = (170,120,120)
        self.high_red_hsv = (180,255,255)
        self.low_yellow_hsv  = (15,110,110)
        self.high_yellow_hsv = (30,255,255)
        self.low_green_hsv  = (80,130,130)
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
            import hilens
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
            # try: 
                w, h = template.shape[::-1]
                w += 4; h += 4 
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where( res >= self.threshold)

                for pt in zip(*loc[::-1]): 
                    try: 
                        pt = (pt[0] - 2, pt[1] - 2)  
                        try: 
                            hsv = cv2.cvtColor(self.__detect_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w], cv2.COLOR_BGR2HSV) 
                        except: 
                            w -= 4; h -= 4 
                            pt = (pt[0] + 2, pt[1] + 2) 
                            hsv = cv2.cvtColor(self.__detect_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w], cv2.COLOR_BGR2HSV) 
                        mask_red = cv2.inRange(hsv, self.low_red_hsv, self.high_red_hsv) + cv2.inRange(hsv, (0,120,120), (10,255,255)) 
                        mask_green = cv2.inRange(hsv, self.low_green_hsv, self.high_green_hsv)
                        mask_yellow = cv2.inRange(hsv, self.low_yellow_hsv, self.high_yellow_hsv)
                        mask_white = cv2.threshold(cv2.cvtColor(self.__detect_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w], cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)[1]
                        sums = [ np.sum(mask) for mask in [mask_red, mask_green, mask_yellow] ]
                        max_loc = np.argmax(sums) 
                        if max_loc == 2: 
                            # print(sums[0] /(w-4)/(h-4)/255, sums[2] /(w-4)/(h-4)/255)
                            if sums[0] /w/h/255 > 0.005: 
                                max_loc = 0
                        if sums[max_loc] < (w-4)*(h-4) * 255 * 0.2 or np.sum(mask_white) < (w-4)*(h-4) * 255 * (0.2 if max_loc == 0 else 0.4): 
                            cv2.rectangle(self.__detect_img, pt, (pt[0] + w, pt[1] + h), (125,125,0), 2)
                            continue
                        # print(np.sum(mask_white) /w/h/255)
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
                            if cv2.waitKey(500) == 27: 
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


class ModelInference():
    def __init__(self, model):
        self.__model = model

    def inference(self, img_bgr):
        # print("inference")
        img_preprocess, img_w, img_h = preprocess(img_bgr)
        output = self.__model.infer([img_preprocess.flatten()])
        ##### 4. 结果输出 #####
        bboxes = get_result(output, img_w, img_h)  # 获取检测结果
        img_rgb, labelName, pts1, pts2 = draw_boxes(img_bgr, bboxes)  #
        return labelName, pts1, pts2


class CameraThreadGuard(threading.Thread):
    def __init__(self, video_path="", RECORD=False, HILENS=True):
        threading.Thread.__init__(self)
        self.__cam = Camera(video_path, RECORD, HILENS)
    def run(self):
        global src_img_buff, SHUTDOWN_SIG 
        print("run CameraThreadGuard")
        try:
            while True:
                # print("cam thread running")
                frame = self.__cam.getFrame()
                if frame is not None:
                    src_img_buff.put(frame)

        finally:
            SHUTDOWN_SIG = True 
            print('cam ended')

class PictureProcessorThreadGuard(threading.Thread):
    def __init__(self, model=None, HILENS=True, socket_enable = False, display = None, RECORD=False, TrackBar=False):
        threading.Thread.__init__(self)
        self.__pp = PictureProcessor(HILENS, RECORD, TrackBar)
        self.__model_enable = False
        self.__socket_enable = socket_enable
        self.__display = display
        if model is not None:
            self.__model_enable = True

        self.__model = ModelInference(model)
    
    def run(self):
        global src_img_buff
        global results_buff
        print("run PictureProcessorThreadGuard")
        try: 
            while not SHUTDOWN_SIG:
                # print("pp thread running")
                img_bgr = src_img_buff.get()
                light_label = ""
                model_label = ""
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0

                light_label, dst_img = self.__pp.detect(img_bgr)
                if self.__model_enable and light_label == "":
                    model_label, pt1, pt2 = self.__model.inference(img_bgr)
                    if model_label != "":
                        xmin = pt1[0]
                        ymin = pt1[1]
                        xmax = pt2[0]
                        ymax = pt2[1]

                result_label = model_label if light_label == "" else light_label

                if self.__socket_enable:
                        results_buff.put((result_label, xmax, xmin, ymax, ymin))

                if self.__display is not None:
                    import hilens
                    h,w,c = img_bgr.shape
                    cv2.putText(img_bgr, result_label, (int(w/2),int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    output_yuv = hilens.cvt_color(img_bgr, hilens.BGR2YUV_NV21)
                    self.__display.show(output_yuv)  # 显示到屏幕上


        finally: 
            print('pp ended') 

class SocketThreadGuard(threading.Thread):
    def __init__(self, socket_enable = False):
        threading.Thread.__init__(self)
        self.__socket_enable = socket_enable

        HOST = ''
        PORT = 7777
        bufsize = 1024
        if self.__socket_enable:
            self.socket_3399 = socket(AF_INET, SOCK_STREAM)
            # self.socket_3399.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_3399.settimeout(1)#timeout 1 second
            self.socket_3399.bind((HOST, PORT))
            self.socket_3399.listen()
            print("socket create successfully")
    
    def run(self):
        global results_buff
        print("run SocketThreadGuard")
        if self.__socket_enable:
            try: 
                while not SHUTDOWN_SIG:
                    print("socket sending msg")
                    result = results_buff.get()
                    msg = str(result[0]) + "," + str(result[1]) + "," + str(result[2]) + "," + str(result[3]) + "," + str(result[4])
                    print(msg)
                    socketSendMsg(self.socket_3399, msg)
                        
            finally: 
                self.socket_3399.close()
                print('socket ended') 

        return

    


if __name__ == "__main__":
    video_path = "src_output3.mp4"
    # video_path = "C:\\Users\\CHKB\\Desktop\\src_output2.avi"
    video_path = "E:\\aboutme\\huawei_self_driving\\videos\\traffic_light\\src_output.mp4"

    cam_thread = CameraThreadGuard(video_path=video_path, RECORD=False, HILENS=False)
    pp_thread = PictureProcessorThreadGuard(model=None, HILENS=False, socket_enable = True, display = None, RECORD=False, TrackBar=False)

    thread_lists = [cam_thread, pp_thread]

    for t in thread_lists:
        t.setDaemon(True)
    print("start")
    try:
        for t in thread_lists:
            t.start()

        # Wait for complete. 
        
        while True:
            time.sleep(1)


        for t in thread_lists:
            t.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        cv2.destroyAllWindows() 
        print('exit')
