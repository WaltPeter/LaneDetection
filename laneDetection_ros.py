#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import datetime
import math
import time
import threading

# define

ROS = True  # Enable ROS communication?
RECORD = False  # Record videos?
SHOW = False  # Show result?
PYTHON2 = True

if PYTHON2:
    from Queue import Queue
else:
    from queue import Queue


SHUTDOWN_SIG = False

LANE_UNDETECTED = 0
LANE_DETECTED = 1

LANE_TURN_LEFT                  = 1
LANE_TURN_RIGHT                   = 2
CIRCLE_TURN         = LANE_TURN_LEFT

SPEED_LIMITED = 1
SPEED_UNLIMITED = 0
IS_SPEED_LIMITED = SPEED_LIMITED

# ifdef
if ROS:
    import rospy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Int32
    from cv_bridge import CvBridge
# endif
def lane_directionJudgecallback(msg):
    global CIRCLE_TURN
    CIRCLE_TURN = msg.data

def speedlimitedwithlaneJudgecallback(msg):
    global IS_SPEED_LIMITED
    IS_SPEED_LIMITED = msg.data

class KalmanFilter:
    def __init__(self):
        self._kalman = cv2.KalmanFilter(2, 1, 0)
        self._kalman.transitionMatrix = np.array(
            [[1., 1.], [0., 1.]], dtype=np.float32)
        self._kalman.measurementMatrix = 1. * np.ones((1, 2), dtype=np.float32)
        self._kalman.processNoiseCov = 1e-5 * np.eye(2, dtype=np.float32)
        self._kalman.measurementNoiseCov = 5e-3 * \
            np.ones((1, 1), dtype=np.float32)
        self._kalman.errorCovPost = 1. * np.ones((2, 2), dtype=np.float32)
        self._kalman.statePost = 0.1 * np.random.randn(2, 1).astype(np.float32)

    def update(self, value):
        value = np.reshape(value, (1, 1)).astype(np.float32)
        self._kalman.correct(value)

    def predict(self):
        return self._kalman.predict()[0, 0]


class AimPoint:
    def __init__(self, x):
        self.current = x
        self.prev = x

    def update(self, new_x):
        self.prev = self.current
        self.current = (self.prev + new_x) / 2

    def predict(self):
        return self.current


class VideoRecorder:
    def __init__(self, name="", fps=24):
        self.name = "./" + name + \
            str(datetime.datetime.now()).replace(":", ".") + ".avi"
        self.writer = None
        self.fps = fps

    def write(self, img):
        if self.writer is None:
            width = img.shape[1]
            height = img.shape[0]
            self.writer = cv2.VideoWriter(self.name,
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (width, height))
        if img is not None:
            self.writer.write(img)

    def release(self):
        self.writer.release()


class Camera:

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if RECORD:
            fps = 20
            size = (1280, 720)
            format = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.src_video_writer = cv2.VideoWriter(
                "/home/pi/Videos/src_output.avi", format, fps, size)

    def getFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            frame = None

        ret = frame

        if frame is not None and RECORD:
            self.src_video_writer.write(frame)

        return ret

    def display(self, frame):
        cv2.imshow("results", frame)
        cv2.waitKey(1)

    def __del__(self):
        self.cap.release()

    def close(self):
        self.__del__()


class History: 
    def __init__(self): 
        self._hist_length = 3 
        self._hist = [(0, False) for i in range(3)] 

    def update(self, coefficient, detected): 
        new_coefficient = coefficient 
        new_detected = detected 
        
        if True in [a[1] for a in self._hist] and not detected: 
            idx = [i for i in range(self._hist_length) if self._hist[i][1]][-1]
            new_coefficient = self._hist[idx][0] 
            new_detected = True 

        self._hist.pop(0) 
        self._hist.append((coefficient, detected)) 

        return new_coefficient, new_detected 

    def getPolyFit(self): 
        x = [i for i in range(self._hist_length)] 
        y = [a[0] for a in self._hist] 
        a, b, c = np.polyfit(x, y, 2) 
        nx = 4 
        return a* nx**2 + b* nx + c 


class PictureProcessor:
    def __init__(self, path="../src2.mp4", record=False):
        self.__fps_start = time.time()
        # TODO: Tune parameter to optimum, when the pedestrian crossing is nearly 20cm from car.
        self.targetDistance = 100
        self.__fps = 0
        self.pedestrianFound = False
        self.isPedestrianTarget = False
        self.particles = list()
        self.kalman = KalmanFilter()
        self.aimKalman = AimPoint(1280/2)
        self.historyFilter = History() 
        self.maxKalmanIdle = 5
        self.kalmanIdle = 0
        self._prev_angle = 0

        # ifdef ROS
        if ROS:
            self.imagePub = rospy.Publisher('images', Image, queue_size=1)
            self.cmdPub = rospy.Publisher('lane_vel', Twist, queue_size=1)
            self.laneJudgePub = rospy.Publisher(
                'laneJudge', Int32, queue_size=1)
            self.pedestrianJudgePub = rospy.Publisher(
                'pedestrianJudge', Int32, queue_size=1)
            rospy.Subscriber("/lane_directionJudge", Int32,lane_directionJudgecallback,queue_size=1)
            rospy.Subscriber("/speed_limited_with_laneJudge", Int32,speedlimitedwithlaneJudgecallback, queue_size=1 )
            self.laneJudge = LANE_UNDETECTED
            self.cam_cmd = Twist()
            self.cvb = CvBridge()
        # endif

    def __del__(self):
        print("laneDetection quitting ...")

    def getParticles(self, num_particles=10, padding_top=50, size=60):
        # @param: num_particles: int: max num of particles.
        # @param: padding_top: int: verticle spacing from top to the 1st particle.

        # Remove old.
        self.particles = list()
        self.pedestrianBndbox = [
            [self.img.shape[1], self.img.shape[0]], [0, 0]]
        self.pedestrianFound = False

        for i in range(num_particles):
            # Get horizontal argmax from window.
            window = self.binary[padding_top+i*size-5:padding_top+i*size+5, :]
            if np.sum(window) < window.shape[0] * window.shape[1] * 255 * 0.02:
                continue
            histogram_x = np.sum(window, axis=0)
            max_x = np.argwhere(histogram_x == np.amax(histogram_x))
            midpoints = self._autoClustering(max_x)

            # Update pedestrian bndbox.
            if self.isPedestrian(max_x, midpoints, window):
                pt = [[np.amin(max_x), padding_top+i*size-20],
                      [np.amax(max_x), padding_top+i*size+20]]
                if pt[0][0] < self.pedestrianBndbox[0][0]:
                    self.pedestrianBndbox[0][0] = pt[0][0]
                if pt[1][0] > self.pedestrianBndbox[1][0]:
                    self.pedestrianBndbox[1][0] = pt[1][0]
                if pt[0][1] < self.pedestrianBndbox[0][1]:
                    self.pedestrianBndbox[0][1] = pt[0][1]
                if pt[1][1] > self.pedestrianBndbox[1][1]:
                    self.pedestrianBndbox[1][1] = pt[1][1]
                self.pedestrianFound = True

            for j, midpoint in enumerate(midpoints):
                # Update point coordinate.
                pt = (midpoint, padding_top+i*size)
                self.particles.append(pt)

    def _autoClustering(self, iterable):
        clusters = list()
        temp_c = list()
        mids = list()
        for i, value in enumerate(iterable):
            if i == 0:
                continue
            m = value - iterable[i-1]
            if m > 100:  # new cluster
                clusters.append(temp_c)
                temp_c = list()
            temp_c.append(value)
        clusters.append(temp_c)
        for cluster in clusters:
            try:
                mids.append(int(np.mean(cluster)))
            except:
                pass  # Except for empty list
        return mids

    def isPedestrian(self, iterable, midpoints, roi):
        status = np.std(iterable) > 100 \
            and ((np.sum(roi) > (np.amax(iterable) - np.amin(iterable)) * roi.shape[0] * 255 * 0.5)
                 or len(midpoints) >= 3)
        return status

    def sortAndFilterParticlesByContours(self):

        # Only detect the outer lane if there is pedestrian crossing.
        temp_ = list()
        filtered_particles = list()
        if self.pedestrianFound:

            bin_inv = 255 - self.binary
            _, contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            rects = list() 
            for contour in contours: 
                center, size, angle = cv2.minAreaRect(contour) 
                if size[0] * size[1] > 0.3 * self.binary.shape[0] * self.binary.shape[1] or \
                    size[0] * size[1] < 0.01 * self.binary.shape[0] * self.binary.shape[1]: 
                    continue 
                rects.append((center, size, angle)) 
            rects = sorted(rects, key=lambda x:x[2]) 
            start = -1; end = len(rects)-1 
            for i, rect in enumerate(rects): 
                if i == 0: continue 
                if abs(rect[2] - rects[i-1][2]) > 30: 
                    if start == -1: start = i 
                    else: end = i; break 
            
            if len(rects) > 0: 
                start = 0 if start == -1 else start 
                rects = sorted(rects[start:end+1], key=lambda x:x[0][0]) 
                box_left = np.int0(cv2.boxPoints(rects[0]))
                box_right = np.int0(cv2.boxPoints(rects[-1]))
                arg_left_x = np.argmin([pt[0] for pt in box_left]) 
                arg_right_x = np.argmax([pt[0] for pt in box_right]) 
                left_x = box_left[arg_left_x][0] 
                right_x = box_right[arg_right_x][0] 
                low_y = np.max((box_left[arg_left_x][1], box_right[arg_right_x][1])) 
            else: 
                left_x = 0 
                right_x = 0 
                low_y = self.img.shape[0] 

            # for rect in rects: 
            #     box = cv2.boxPoints(rect) 
            #     box = np.int0(box)
            #     cv2.drawContours(bin_inv, [box], 0, (125), 5) 
            # cv2.imshow("cont", bin_inv)
            # cv2.waitKey(0)

            for a, particle in enumerate(self.particles):
                if len(temp_) == 0:
                    temp_.append(particle)
                    continue
                if particle[0] > left_x and particle[0] < right_x and particle[1] > low_y: continue 
                if not temp_[-1][1] == particle[1] or a+1 >= len(self.particles):
                    if a+1 >= len(self.particles):
                        temp_.append(particle)
                    # Get the left most lane.
                    if self._prev_angle > 0:
                        idx = np.argwhere([p[0] for p in temp_] == np.amin(
                            [p[0] for p in temp_]))[0][0]
                    # Get the right most lane.
                    else:
                        idx = np.argwhere([p[0] for p in temp_] == np.amax(
                            [p[0] for p in temp_]))[0][0]
                    filtered_particles.append(temp_[idx])
                    temp_ = [particle]
                else:
                    temp_.append(particle)
            self.particles = filtered_particles

        # Sort particles by contours.

        # @struct: dict<list: idx of particle>: key=idx of contour
        self.groupedParticles = list()

        # Find all white contours in self.binary.
        _, self.contours, self.hierarchy = cv2.findContours(
            self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # For every contour, find all particles in that contour and sorted in a dict of list.
        self.particles = sorted(
            [p for p in self.particles if p is not None], key=lambda x: x[1])
        for i, contour in enumerate(self.contours):
            temp_ = list()
            for particle in self.particles:
                # Is this particle in this contour.
                distance = cv2.pointPolygonTest(contour, particle, True)
                if distance >= 0:
                    temp_.append(particle)
            # if len(temp_) > 3: 
            #     xs = [a[0] for a in temp_] 
            #     ys = [a[1] for a in temp_] 
            #     p = np.polynomial.Polynomial.fit(xs, ys, 2) 
            #     new_ys = [int(150+i*50) for i in range(10)] 
            #     new_xs = [int((p-y).roots()) for y in new_ys] 
            #     temp_ = [(new_xs[i], new_ys[i]) for i in range(len(new_xs))] 
            self.groupedParticles.append(temp_)

        # Sort groups by num of particle and get maximum 2 groups.
        self.groupedParticles = sorted(
            self.groupedParticles, key=lambda x: len(x), reverse=True)[:2]

        # Filter particles.
        colors = [(100, 100, 255), (255, 100, 100)]
        filtered_particles = list()
        for i, group in enumerate(self.groupedParticles):
            if len(group) < 3:
                continue
            filtered_particles.append(group)
            # Visualize.
            for particle in group:
                cv2.circle(self.img, particle, 10, colors[i], -1)
        self.groupedParticles = filtered_particles
        self.particles = [p for group in self.groupedParticles for p in group]

    def __calCoefficient(self, coefficient):
        global CIRCLE_TURN
        global IS_SPEED_LIMITED
        tmp = coefficient
        #CIRCLE_TURN = LANE_TURN_LEFT
        if IS_SPEED_LIMITED == SPEED_LIMITED:
            if abs(coefficient) < 0.55: 
                if abs(coefficient) < 0.45:
                    tmp = coefficient * 0.4
                else:
                    tmp = coefficient * 0.8
            else:
                if abs(coefficient) < 0.7:
                    tmp = coefficient * 1
                else:
                    tmp = coefficient * 1.2

        else:
            if CIRCLE_TURN == LANE_TURN_LEFT:
                # Manual multiplier.
                print("turn left")
                if abs(coefficient) < 0.55:
                    if abs(coefficient) < 0.45:
                        tmp = coefficient * 1
                        if coefficient > 0:
                            tmp = coefficient * 0.1

                    else:
                        tmp = coefficient * 1.4
                        if coefficient > 0:
                            tmp = coefficient * 0.8
                else:
                    if abs(coefficient) < 0.7:
                        tmp = coefficient * 1.3
                        if coefficient > 0:
                            tmp = coefficient * 1
                    else:
                        tmp = coefficient * 1.3
                        if coefficient > 0:
                            tmp = coefficient * 1

            elif CIRCLE_TURN == LANE_TURN_RIGHT:
                print("turn right")
                if abs(coefficient) < 0.55:
                    if abs(coefficient) < 0.45:
                        tmp = coefficient * 0.65
                        if coefficient < 0:
                            tmp = coefficient * 0.2
                    else:
                        tmp = coefficient * 1
                        if coefficient < 0:
                            tmp = coefficient * 0.5
                else:
                    if abs(coefficient) < 0.7:
                        tmp = coefficient * 1.1
                    else:
                        tmp = coefficient * 1.2

            
        
        # @return range [-1, 1]
        tmp = max(-1, tmp) if tmp < 0 else min(1, tmp)
        return tmp * (-15)

    def decision(self):
        # if self.pedestrianFound: self._pedestrianSegmentation()
        self.sortAndFilterParticlesByContours()

        # Dynomic aim point.
        cx = int(self.aimKalman.predict())  # int(self.img.shape[1]/2)
        cy = int(self.img.shape[0]/2)
        key_point = list()
        cv2.circle(
            self.img, (int(self.img.shape[1]/2), cy), 10, (255, 255, 255), -1)
        cv2.circle(self.img, (cx, cy), 10, (0, 255, 0), 2)

        # No result.
        if len(self.groupedParticles) == 0:
            coefficient, detected = self.historyFilter.update(0, False) 
            self.kalman.update(0)
            self.aimKalman.update(self.img.shape[1]/2)
            cv2.putText(self.img, "%.4f %s" % (coefficient, "<-" if coefficient < 0 else "->"), (cx-100, cy-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
            return detected, coefficient, self.isPedestrianTarget

        # if self.pedestrianFound: 
        #     coefficient = self.historyFilter.getPolyFit() 
        #     cv2.putText(self.img, "%.4f %s" % (coefficient, "<-" if coefficient < 0 else "->"), (cx-100, cy-15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 255), 2)
        #     return True, coefficient, True 

        self.kalmanIdle = 0

        for group in self.groupedParticles:
            x_vals = np.array([p[0] for p in group])
            key_point.append(np.average(x_vals))
            cv2.line(
                self.img, (int(key_point[-1]), cy), (cx, cy), (0, 255, 0), 5)
            cv2.circle(
                self.img, (int(key_point[-1]), cy), 10, (100, 255, 100), -1)
            cv2.circle(self.img, tuple(group[-1]), 10, (0, 0, 0), -1)

        idxs = [0, 1]
        if len(key_point) >= 2:
            if key_point[1] < key_point[0]:
                idx = [1, 0]
            left_lane, right_lane = sorted(key_point[:2])
        else:
            if key_point[0] < cx:
                left_lane = key_point[0]
                right_lane = self.img.shape[1] + cx
            else:
                left_lane = -cx
                right_lane = key_point[0]

        # Calculate distance.
        dist_left = abs(cx - left_lane)
        dist_right = abs(right_lane - cx)
        dist_total = dist_left + dist_right

        # Calculate steer angle coefficient.
        l_ratio = dist_left / dist_total
        r_ratio = dist_right / dist_total
        # @return range [-1, 1]
        coefficient = max(r_ratio, l_ratio) - \
            0.5 if r_ratio > l_ratio else -max(r_ratio, l_ratio)+0.5
        coefficient *= 2

        self.historyFilter.update(coefficient, True)

        # Overboundary check.
        self.isOverboundary = False
        if abs(coefficient) > 0.5:
            if len(key_point) >= 2:
                if coefficient < 0:
                    if self.groupedParticles[idxs[1]][-1][0] < cx:
                        self.isOverboundary = True
                else:
                    if self.groupedParticles[idxs[0]][-1][0] > cx:
                        self.isOverboundary = True
            else:
                if coefficient < 0:
                    if self.groupedParticles[0][-1][0] < cx:
                        self.isOverboundary = True
                else:
                    if self.groupedParticles[0][-1][0] > cx:
                        self.isOverboundary = True
        if self.isOverboundary and not self.pedestrianFound:
            coefficient = -coefficient / abs(coefficient)
            cv2.line(
                self.img, (int(key_point[-1]), cy), (cx, cy), (255, 100, 255), 5)
            cv2.circle(
                self.img, (int(key_point[-1]), cy), 10, (255, 100, 255), -1)
            cv2.circle(
                self.img, (int(key_point[-1]), cy), 10, (255, 100, 255), -1)
            cv2.putText(self.img, "Overboundary", (cx-100, cy+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 3)

        if not self.pedestrianFound:
            self.kalman.update(coefficient)
            self._prev_angle = coefficient = self.kalman.predict()
            self.aimKalman.update(
                self.img.shape[1]/2 + math.pow(coefficient, 3) * 200)

        # Check pedestrian is at 20cm distance.
        if self.pedestrianFound:
            # pedestrian height.  
            ty = abs(self.pedestrianBndbox[0][1] -
                  self.pedestrianBndbox[1][1]) 
            self.isPedestrianTarget = ty > self.targetDistance
        else:
            self.isPedestrianTarget = False

        # Visualize.
        if self.pedestrianFound:
            if self.isPedestrianTarget:
                color = (255, 153, 0)
            else:
                color = (0, 255, 255)
            cv2.rectangle(self.img, tuple(self.pedestrianBndbox[0]), tuple(
                self.pedestrianBndbox[1]), color, 10)

        cv2.putText(self.img, "%.4f %s" % (coefficient, "<-" if coefficient < 0 else "->"), (cx-100, cy-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return True, coefficient, self.isPedestrianTarget

    def detectLane(self, img):

        # Grab raw image.
        self.img = img
        if img is None:
            return False

        self.img = cv2.resize(
            self.img[480:], (self.img.shape[1], self.img.shape[0]))

        # Preprocessing.
        # self.binary = cv2.threshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)[1]
        self.binary = cv2.inRange(cv2.cvtColor(
            self.img, cv2.COLOR_BGR2HSV), (0, 0, 135), (180, 30, 255))

        self.getParticles()

        isLaneDetected, coefficient, isPedestrianTarget = self.decision()
        # try: self.dst_wr.write(self.img)
        # except: pass # Record dst video.

        # ifdef ROS
        if ROS:
            self.laneJudge = LANE_DETECTED if isLaneDetected else LANE_UNDETECTED
            self.cam_cmd.angular.z = self.__calCoefficient(coefficient)
            #print("angular.z:   ",self.cam_cmd.angular.z)
            self.laneJudgePub.publish(self.laneJudge)
            self.pedestrianJudgePub.publish(1 if isPedestrianTarget else 0)
            self.cmdPub.publish(self.cam_cmd)
        # endif

        if SHOW:
            cv2.imshow("thresh", self.binary)
            cv2.imshow("result", self.img)
            cv2.waitKey(1)

        # ifdef ROS

        if self.__fps == 30:
            duration = time.time() - self.__fps_start
            print("fps", self.__fps/duration)
            self.__fps = 0
            self.__fps_start = time.time()
        else:
            self.__fps = self.__fps + 1

        return True

    def close(self):
        self.__del__()


class CameraThreadGuard(threading.Thread):
    def __init__(self, video_path=""):
        threading.Thread.__init__(self)
        self.__cam = Camera(video_path)

    def run(self):
        global src_img_buff, SHUTDOWN_SIG
        try:
            while True:
                frame = self.__cam.getFrame()
                if frame is not None:
                    src_img_buff.put(frame)
                else:
                    self.__cam.close()
                    break
        finally:
            SHUTDOWN_SIG = True
            print('cam ended')


class PictureProcessorThreadGuard(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__pp = PictureProcessor()

    def run(self):
        global src_img_buff, SHUTDOWN_SIG
        # ifdef
        if ROS:

            try:
                while not rospy.is_shutdown():
                    self.__pp.detectLane(src_img_buff.get())
                    rate.sleep()
            except rospy.ROSInterruptException:
                SHUTDOWN_SIG = True
                self.__pp.close()
                print(rospy.ROSInterruptException)
                pass
        # else
        else:
            # try:
                while not SHUTDOWN_SIG:
                    self.__pp.detectLane(src_img_buff.get())
                cv2.destroyAllWindows()
            # except Exception as e: 
            #     print(e)
        # endif
        print('pp ended')


if __name__ == "__main__":
    if ROS:
        rospy.init_node("lane_vel", anonymous=True)
        rate = rospy.Rate(30)
    src_img_buff = Queue(1)
    # video_path = "30.mp4"
    # video_path = "E:\\aboutme\\huawei_self_driving\\videos\\lane\\35.mp4"
    video_path = "/dev/video10"

    cam_thread = CameraThreadGuard(video_path=video_path)
    pp_thread = PictureProcessorThreadGuard()

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
        SHUTDOWN_SIG = False
        print("KeyboardInterrupt")
        cv2.destroyAllWindows()
        print('exit')

    print("End:0")




