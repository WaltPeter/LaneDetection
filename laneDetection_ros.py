#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2 
import numpy as np 
import datetime 
import math 
from time import time 

# define 
ROS = True  # Enable ROS communication? 
RECORD = True  # Record videos? 
SHOW = False  # Show result? 

LANE_UNDETECTED = 0
LANE_DETECTED = 1

# ifdef 
if ROS: 
    import rospy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Int32
    from cv_bridge import CvBridge
# endif 


class KalmanFilter: 
    def __init__(self): 
        self._kalman = cv2.KalmanFilter(2, 1, 0)
        self._kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]], dtype=np.float32)
        self._kalman.measurementMatrix = 1. * np.ones((1, 2), dtype=np.float32)
        self._kalman.processNoiseCov = 1e-5 * np.eye(2, dtype=np.float32)
        self._kalman.measurementNoiseCov = 5e-3 * np.ones((1, 1), dtype=np.float32)
        self._kalman.errorCovPost = 1. * np.ones((2, 2), dtype=np.float32)
        self._kalman.statePost = 0.1 * np.random.randn(2, 1).astype(np.float32) 

    def update(self, value):
        value = np.reshape(value, (1,1)).astype(np.float32) 
        self._kalman.correct(value)

    def predict(self): 
        return self._kalman.predict()[0,0]

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
        self.name = "./" + name + str(datetime.datetime.now()).replace(":", ".") + ".avi"
        self.writer = None 
        self.fps = fps 
    
    def write(self, img): 
        if self.writer is None: 
            width = img.shape[1]; height = img.shape[0]
            self.writer = cv2.VideoWriter(self.name, 
                        cv2.VideoWriter_fourcc('M','J','P','G'), self.fps, (width, height))
        if img is not None: 
            self.writer.write(img)

    def release(self): 
        self.writer.release() 


class Camera:  
    def __init__(self, path="../src2.mp4", record=False):

        self.targetDistance = 300  # TODO: Tune parameter to optimum, when the pedestrian crossing is nearly 20cm from car. 

        self.pedestrianFound = False 
        self.isPedestrianTarget = False 
        self.cap = cv2.VideoCapture(path) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
        self.particles = list() 
        self.kalman = KalmanFilter() 
        self.aimKalman = AimPoint(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2) 
        self.maxKalmanIdle = 5 
        self.kalmanIdle = 0 
        self._prev_angle = 0 
        if record: 
            self.src_wr = VideoRecorder("SRC_")
            self.dst_wr = VideoRecorder("DST_") 

        # ifdef ROS 
        if ROS: 
            self.imagePub = rospy.Publisher('images', Image, queue_size=1)
            self.cmdPub = rospy.Publisher('lane_vel', Twist, queue_size=1)
            self.laneJudgePub = rospy.Publisher('laneJudge',Int32,queue_size=1)
            self.pedestrianJudgePub = rospy.Publisher('pedestrianJudge',Int32,queue_size=1)
            self.laneJudge = LANE_UNDETECTED
            self.cam_cmd = Twist()
            self.cvb = CvBridge()
        # endif 

    def __del__(self):
        self.cap.release()
        try: 
            self.src_wr.release() 
            self.dst_wr.release() 
        except: pass 

    def getParticles(self, num_particles=14, padding_top=20, size=50): 
        # @param: num_particles: int: max num of particles. 
        # @param: padding_top: int: verticle spacing from top to the 1st particle. 

        # Remove old. 
        self.particles = list() 
        self.pedestrianBndbox = [[self.img.shape[1], self.img.shape[0]], [0, 0]]
        self.pedestrianFound = False 

        for i in range(num_particles): 
            # Get horizontal argmax from window. 
            window = self.binary[padding_top+i*size-20:padding_top+i*size+20, :]
            if np.sum(window) < window.shape[0] * window.shape[1] * 255 * 0.02: continue 
            histogram_x = np.sum(window, axis=0) 
            max_x = np.argwhere(histogram_x == np.amax(histogram_x)) 
            midpoints = self._autoClustering(max_x) 

            # Update pedestrian bndbox. 
            if self.isPedestrian(max_x, midpoints, window): 
                pt = [[np.amin(max_x), padding_top+i*size-20], [np.amax(max_x), padding_top+i*size+20]]
                if pt[0][0] < self.pedestrianBndbox[0][0]: self.pedestrianBndbox[0][0] = pt[0][0] 
                if pt[1][0] > self.pedestrianBndbox[1][0]: self.pedestrianBndbox[1][0] = pt[1][0] 
                if pt[0][1] < self.pedestrianBndbox[0][1]: self.pedestrianBndbox[0][1] = pt[0][1] 
                if pt[1][1] > self.pedestrianBndbox[1][1]: self.pedestrianBndbox[1][1] = pt[1][1] 
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
            if i == 0: continue 
            m = value - iterable[i-1] 
            if m > 100: # new cluster 
                clusters.append(temp_c) 
                temp_c = list() 
            temp_c.append(value) 
        clusters.append(temp_c) 
        for cluster in clusters: 
            try: 
                mids.append(int(np.mean(cluster))) 
            except: pass # Except for empty list 
        return mids  

    def isPedestrian(self, iterable, midpoints, roi): 
        status = np.std(iterable) > 100 \
            and ( (np.sum(roi) > (np.amax(iterable) - np.amin(iterable)) * roi.shape[0] * 255* 0.5) \
            or len(midpoints) >= 3)
        return status

    def sortAndFilterParticlesByContours(self): 

        # Only detect the outer lane if there is pedestrian crossing. 
        temp_ = list() 
        filtered_particles = list() 
        if self.pedestrianFound: 
            for a, particle in enumerate(self.particles): 
                if len(temp_) == 0: 
                    temp_.append(particle) 
                    continue 
                if not temp_[-1][1] == particle[1] or a+1 >= len(self.particles): 
                    if a+1 >= len(self.particles): temp_.append(particle) 
                    # Get the left most lane. 
                    if self._prev_angle > 0: 
                        idx = np.argwhere([p[0] for p in temp_] == np.amin([p[0] for p in temp_]))[0][0] 
                    # Get the right most lane. 
                    else: 
                        idx = np.argwhere([p[0] for p in temp_] == np.amax([p[0] for p in temp_]))[0][0] 
                    filtered_particles.append(temp_[idx]) 
                    temp_ = [particle] 
                else: 
                    temp_.append(particle)
            self.particles = filtered_particles 

        # Sort particles by contours. 

        # @struct: dict<list: idx of particle>: key=idx of contour
        self.groupedParticles = list() 

        # Find all white contours in self.binary. 
        _, self.contours, self.hierarchy = cv2.findContours(self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 

        # For every contour, find all particles in that contour and sorted in a dict of list. 
        self.particles = sorted([p for p in self.particles if p is not None], key=lambda x: x[1]) 
        for i, contour in enumerate(self.contours): 
            temp_ = list() 
            for particle in self.particles: 
                # Is this particle in this contour. 
                distance = cv2.pointPolygonTest(contour, particle, True) 
                if distance >= 0: temp_.append(particle) 
            self.groupedParticles.append(temp_) 

        # Sort groups by num of particle and get maximum 2 groups. 
        self.groupedParticles = sorted(self.groupedParticles, key=lambda x: len(x), reverse=True)[:2]  

        # Filter particles. 
        colors = [(100,100,255), (255,100,100)]
        filtered_particles = list() 
        for i, group in enumerate(self.groupedParticles): 
            if len(group) < 3: continue 
            filtered_particles.append(group) 
            # Visualize. 
            for particle in group: 
                cv2.circle(self.img, particle, 10, colors[i], -1) 
        self.groupedParticles = filtered_particles 
        self.particles = [p for group in self.groupedParticles for p in group] 

    def decision(self): 
        # if self.pedestrianFound: self._pedestrianSegmentation() 
        self.sortAndFilterParticlesByContours() 

        # Dynomic aim point. 
        cx = int(self.aimKalman.predict()) # int(self.img.shape[1]/2) 
        cy = int(self.img.shape[0]/2) 
        key_point = list() 
        cv2.circle(self.img, (int(self.img.shape[1]/2), cy), 10, (255,255,255), -1) 
        cv2.circle(self.img, (cx, cy), 10, (0,255,0), 2) 

        # No result. 
        if len(self.groupedParticles) == 0: 
            self.kalman.update(0) 
            self.aimKalman.update(self.img.shape[1]/2) 
            if self.kalmanIdle < self.maxKalmanIdle: 
                coefficient = self.kalman.predict() 
                self.kalmanIdle += 1 
                return True, coefficient, self.isPedestrianTarget 
            else: 
                return False, 0, self.isPedestrianTarget
        
        self.kalmanIdle = 0 

        for group in self.groupedParticles: 
            x_vals = np.array([p[0] for p in group])
            key_point.append( np.average(x_vals) ) 
            cv2.line(self.img, (int(key_point[-1]), cy), (cx, cy), (0,255,0), 5) 
            cv2.circle(self.img, (int(key_point[-1]), cy), 10, (100,255,100), -1) 
            cv2.circle(self.img, tuple(group[-1]), 10, (0,0,0), -1) 

        idxs = [0,1]
        if len(key_point) >= 2: 
            if key_point[1] < key_point[0]: idx = [1,0]
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
        coefficient = max(r_ratio, l_ratio)-0.5 if r_ratio > l_ratio else -max(r_ratio, l_ratio)+0.5 
        coefficient *= 2 

        # Overboundary check. 
        self.isOverboundary = False 
        if abs(coefficient) > 0.5: 
            if len(key_point) >= 2: 
                if coefficient < 0: 
                    if self.groupedParticles[idxs[1]][-1][0] < cx: self.isOverboundary = True 
                else: 
                    if self.groupedParticles[idxs[0]][-1][0] > cx: self.isOverboundary = True 
            else: 
                if coefficient < 0: 
                    if self.groupedParticles[0][-1][0] < cx: self.isOverboundary = True 
                else: 
                    if self.groupedParticles[0][-1][0] > cx: self.isOverboundary = True 
        if self.isOverboundary: 
            coefficient = -coefficient / abs(coefficient) 
            cv2.line(self.img, (int(key_point[-1]), cy), (cx, cy), (255,100,255), 5) 
            cv2.circle(self.img, (int(key_point[-1]), cy), 10, (255,100,255), -1) 
            cv2.circle(self.img, (int(key_point[-1]), cy), 10, (255,100,255), -1) 
            cv2.putText(self.img, "Overboundary", (cx-100, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,255), 3)

        if not self.pedestrianFound: 
            self.kalman.update(coefficient) 
            self._prev_angle = self.kalman.predict() 
        self.aimKalman.update(self.img.shape[1]/2 + math.pow(coefficient, 3) * 200) 

        # Check pedestrian is at 20cm distance. 
        if self.pedestrianFound: 
            ty = (self.pedestrianBndbox[0][1] + self.pedestrianBndbox[1][1]) / 2 
            self.isPedestrianTarget = abs(ty - self.targetDistance) < 50 
        else: self.isPedestrianTarget = False 
        
        # Visualize. 
        if self.pedestrianFound: 
            if self.isPedestrianTarget: color = (255,153,0) 
            else: color = (0,255,255) 
            cv2.rectangle(self.img, tuple(self.pedestrianBndbox[0]), tuple(self.pedestrianBndbox[1]), color, 10) 

        cv2.putText(self.img, "%.4f %s"%(coefficient, "<-" if coefficient < 0 else "->"), (cx-100, cy-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 

        return True, coefficient, self.isPedestrianTarget
        
    def detectLane(self):
        start = time() 
        
        # Grab raw image. 
        ret, self.img = self.cap.read()
        if not ret: return ret 
        try: self.src_wr.write(self.img) 
        except: pass # Record src video. 

        self.img = cv2.resize(self.img[480:], (self.img.shape[1], self.img.shape[0]))  

        # Preprocessing. 
        # self.binary = cv2.threshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)[1]
        self.binary = cv2.inRange(cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV), (0, 0, 200), (180, 40, 255)) 

        self.getParticles()

        isLaneDetected, coefficient, isPedestrianTarget = self.decision() 
        
        # try: self.dst_wr.write(self.img) 
        # except: pass # Record dst video. 

        # ifdef ROS 
        if ROS: 
            self.laneJudge = LANE_DETECTED if isLaneDetected else LANE_UNDETECTED
            # 强制倍增系数
            tmp = 0
            if abs(coefficient) < 0.5: 
                tmp = coefficient * 1
            else:
                tmp = coefficient * 1.15
            # @return range [-1, 1]
            tmp = max(-1, tmp) if tmp < 0 else min(1, tmp)
            self.cam_cmd.angular.z = -tmp * 14  # TODO: Scale up. 
            self.laneJudgePub.publish(self.laneJudge)
            self.pedestrianJudgePub.publish(1 if isPedestrianTarget else 0) 
            self.cmdPub.publish(self.cam_cmd)
        # endif
        
        if SHOW:
            cv2.imshow("thresh", self.binary)
            cv2.imshow("result", self.img)

        # ifdef ROS 
        if ROS: cv2.waitKey(1)

        duration = time() - start 
        print("fps", 1/duration) 

        return ret 

        
if __name__ == "__main__":

    cam = Camera(path="../../2/2/origin (2).avi", record=RECORD) # TODO: Setup camera with path 
    # cam = Camera(path="/dev/video10", record=RECORD) # TODO: Setup camera with path 

    # ifdef 
    if ROS: 
        rospy.init_node("lane_vel", anonymous=True)
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown():
                cam.detectLane()
                rate.sleep()
        except rospy.ROSInterruptException:
            print(rospy.ROSInterruptException)
            pass
    # else 
    else: 
        while cam.detectLane(): 
            if cv2.waitKey(500) == 27: break  
        cv2.destroyAllWindows() 
    # endif 
       
