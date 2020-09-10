#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2 
import numpy as np 
import datetime 
import math 
from time import time 

# define 
ROS = True  # Enable ROS communication? 
RECORD = False  # Record videos? 

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
        self._kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1), dtype=np.float32)
        self._kalman.errorCovPost = 1. * np.ones((2, 2), dtype=np.float32)
        self._kalman.statePost = 0.1 * np.random.randn(2, 1).astype(np.float32) 

    def update(self, value):
        value = np.reshape(value, (1,1)).astype(np.float32) 
        self._kalman.correct(value)

    def predict(self): 
        return self._kalman.predict()[0,0]


class Particle: 
    def __init__(self, value):
        self.prev = tuple() 
        self.current = value 
    
    def update(self, value, smoothing=True): 
        self.prev = self.current 
        if smoothing: self.current = (int((self.prev[0] + value[0]) / 2), value[1])  
        else: self.current = value 
    

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
        self.pedestrianFound = False 
        self.cap = cv2.VideoCapture(path) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
        self.particles = dict() 
        src_points = np.array([[0,720], [250,560], [966,552], [1280,720]], dtype="float32")
        dst_points = np.array([[300, 686.], [266., 119], [931., 120], [931., 701.]], dtype="float32")
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.kalman = KalmanFilter() 
        if record: 
            self.src_wr = VideoRecorder("SRC_")
            self.dst_wr = VideoRecorder("DST_") 

    def __del__(self):
        self.cap.release()
        try: 
            self.src_wr.release() 
            self.dst_wr.release() 
        except: pass 

    def getParticles(self, num_particles=10, padding_top=20, size=40): 
        # @param: num_particles: int: max num of particles. 
        # @param: padding_top: int: verticle spacing from top to the 1st particle. 

        # Remove old. 
        self.particles = dict() 

        # Pedestrian bndbox. 
        #                         xmin,              ymin,             xmax, ymax
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
                midpoints = [np.amin(midpoints), np.amax(midpoints)] 
                pt = [[np.amin(max_x), padding_top+i*size-20], [np.amax(max_x), padding_top+i*size+20]]
                if pt[0][0] < self.pedestrianBndbox[0][0]: self.pedestrianBndbox[0][0] = pt[0][0] 
                if pt[1][0] > self.pedestrianBndbox[1][0]: self.pedestrianBndbox[1][0] = pt[1][0] 
                if pt[0][1] < self.pedestrianBndbox[0][1]: self.pedestrianBndbox[0][1] = pt[0][1] 
                if pt[1][1] > self.pedestrianBndbox[1][1]: self.pedestrianBndbox[1][1] = pt[1][1] 
                self.pedestrianFound = True 

            for j, midpoint in enumerate(midpoints): 
                # Update point coordinate. 
                pt = (midpoint, padding_top+i*size)
                try: 
                    self.particles[i*10+j].update(pt, False) 
                except: 
                    self.particles[i*10+j] = Particle(pt) 

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

        # @struct: dict<list: idx of particle>: key=idx of contour
        sortParticlesByContours = dict() 

        # Find all white contours in self.binary. 
        _, self.contours, self.hierarchy = cv2.findContours(self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 

        # For every contour, find all particles in that contour and sorted in a dict of list. 
        for j in sorted([k for k in self.particles]): 
            particle = self.particles[j] 
            if particle is None: continue 
            if self.pedestrianFound: 
                if particle.current[0] > self.pedestrianBndbox[0][0] \
                    and particle.current[0] < self.pedestrianBndbox[1][0] \
                    and particle.current[1] > self.pedestrianBndbox[0][1] \
                    and particle.current[1] < self.pedestrianBndbox[1][1]: 
                    continue 
            for i, contour in enumerate(self.contours): 
                distance = cv2.pointPolygonTest(contour, particle.current, True) 
                if distance >= 0: 
                    try: sortParticlesByContours[i].append(j) 
                    except: sortParticlesByContours[i] = [j] 
        
        # Descending sort the dict by num of particles. 
        contoursIndexSortByNumOfParticles = sorted([i for i in sortParticlesByContours], 
                                                    key=lambda x:len(sortParticlesByContours[x]), 
                                                    reverse=True) 
        
        # Get 2 contours with the largest num of particles. 
        contoursIndexSortByNumOfParticles = contoursIndexSortByNumOfParticles[:2] 

        # Filter particles. 
        colors = [(100,100,255), (255,100,100)]
        filtered_particles = dict() 
        self.groupedParticles = list()  
        for i, contourIdx in enumerate(contoursIndexSortByNumOfParticles): 
            if len(sortParticlesByContours[contourIdx]) < 3: continue 
            self.groupedParticles.append(list())  
            for particleIdx in sortParticlesByContours[contourIdx]: 
                filtered_particles[particleIdx] = self.particles[particleIdx]
                self.groupedParticles[i].append(self.particles[particleIdx]) 
                # Visualize. 
                cv2.circle(self.img, self.particles[particleIdx].current, 10, colors[i], -1) 
        self.particles = filtered_particles

        # Visualize. 
        if self.pedestrianFound: 
            cv2.rectangle(self.img, tuple(self.pedestrianBndbox[0]), tuple(self.pedestrianBndbox[1]), (0,255,255), 10) 

    def decision(self): 
        # if self.pedestrianFound: self._pedestrianSegmentation() 
        self.sortAndFilterParticlesByContours() 
        if self.pedestrianFound: 
            confidence_direction_steerAngle = self._getPedestrianGuideLine() 
            if confidence_direction_steerAngle is not None: 
                return confidence_direction_steerAngle 

        confidence = 0 
        group_gradient = list() 

        for group in self.groupedParticles: 
            gradient_list = list() 
            for i, particle in enumerate(group): 
                if particle is None: continue 
                confidence += 1 
                if confidence > 1: 
                    pt2 = particle.current 
                    if group[i-1] is None: continue 
                    pt1 = group[i-1].current 
                    gradient_list.append( (pt2[1] - pt1[1]) / (pt2[0] - pt1[0] + 1e-5) / 720 * 1280 ) 

                    if len(gradient_list) > 1: 
                        if np.std(gradient_list) > 10: 
                            confidence -= 1 
                            gradient_list[-1] += 2 * np.mean(gradient_list[:-2]) 
                            gradient_list[-1] /= 3
            
            # Fit polynomial. 
            if confidence > 0 or len(group) >= 3:  
                # Second degree polynomial. 
                a, b, c = np.polyfit([p.current[0] for p in group], [p.current[1] for p in group], deg=2) 
                # cv2.putText(self.img, str(a), tuple(group[0].current), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2) 
                group_gradient.append( a / abs(a) * np.average(gradient_list)  )
            else: 
                group_gradient.append( np.average(gradient_list)  ) 

        weighted = [len(i) for i in self.groupedParticles] / np.sum([len(i) for i in self.groupedParticles]) 
        avg_gradient = np.sum(np.multiply(group_gradient, weighted))  

        if str(avg_gradient) == "nan": avg_gradient = 0  
        if avg_gradient < 0 and avg_gradient > -1000 and confidence > 0: 
            direction =   1                  # Right
        elif avg_gradient > 0 and avg_gradient < 1000 and confidence > 0: 
            direction =   -1                  # Left
        else: direction = 0; steerAngle = 0   # Straight 

        # Steering Angle (deg)
        steerAngle = math.atan(-avg_gradient) * 180 / math.pi + 90 if confidence > 0 else 0 
        if steerAngle > 90: steerAngle = 180 - steerAngle 
        if steerAngle > 45: steerAngle = 90 - steerAngle 

        steerAngle *= direction 
        if confidence > 0: 
            self.kalman.update(steerAngle) 
            steerAngle = self.kalman.predict()

        if steerAngle > 0: direction = "---->" 
        elif steerAngle < 0: direction = "<----" 
        else: direction = "Straight"

        cv2.putText(self.img, "Confidence: {}".format(confidence), (500,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 
        cv2.putText(self.img, "Direction: {}".format(direction), (500,75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 
        cv2.putText(self.img, "SteerAngle: %.1f deg"%(steerAngle), (500,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 

        return confidence, direction, steerAngle 

    def _pedestrianSegmentation(self): 
        if self.lines is not None: 
            for line in self.lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(self.binary,(x1,y1),(x2,y2),(0),2) 

    def _getPedestrianGuideLine(self): 
        angle_list = list() 
        for i, contour in enumerate(self.contours): 
            hierarchy = self.hierarchy[0][i]
            area = cv2.contourArea(contour) 
            if hierarchy[3] == 0 or area < self.img.shape[0] * self.img.shape[1] * 0.05: continue 
            center, size, angle = cv2.minAreaRect(contour) 
            if area / (size[0]*size[1]) > 0.6: 
                cv2.drawContours(self.img, [contour], -1, (0,255,255), -1) 
                if size[0] < size[1]: angle -= 90 
                angle = abs(angle) 
                if angle >= 90: angle = -(angle - 90) 
                else: angle: 90 - angle 
                angle_list.append(angle) 
        if len(angle_list) > 0: 
            steerAngle = np.mean(angle_list) 
            if steerAngle > 0: direction = "---->" 
            elif steerAngle < 0: direction = "<----" 
            else: direction = "Straight"
            confidence = len(angle_list)
            return confidence, direction, steerAngle 
        else: return None 
        
    def detectLane(self):
        start = time() 
        
        # Grab raw image. 
        ret, self.img = self.cap.read()
        if not ret: return ret 
        try: self.src_wr.write(self.img) 
        except: pass # Record src video. 

        self.img = cv2.resize(self.img[480:], (self.img.shape[1], self.img.shape[0]))  

        # Preprocessing. 
        # self.img = cv2.warpPerspective(self.img, self.M, (1280, 720), cv2.INTER_LINEAR)
        edges = cv2.Canny(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=3)
        self.lines = cv2.HoughLines(edges, 1, np.pi/180, 175 if np.sum(self.img) > 75000000 else 200)

        self.binary = cv2.threshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)[1]

        self.getParticles()

        detectionConfident, driveDirection, steerAngle = self.decision() 
        cv2.imshow("thresh", self.binary) 
        cv2.imshow("result", self.img) 
        try: self.dst_wr.write(self.img) 
        except: pass # Record dst video. 

        # ifdef ROS 
        if ROS: 
            self.laneJudge = LANE_DETECTED
            self.cam_cmd.angular.z = steerAngle
            self.laneJudgePub.publish(self.laneJudge)
            self.cmdPub.publish(self.cam_cmd)
            self.imagePub.publish(self.cvb.cv2_to_imgmsg(self.binary))  # self.binary
        # endif

        duration = time() - start 
        print("fps", 1/duration) 
        
        return ret 
        
        cv2.imshow("thresh", self.binary)
        cv2.imshow("result", self.img)
        self.out2.write(self.img) 
        cv2.waitKey(1)
        
        duration = time() - start 
        print("fps", 1/duration) 

        
if __name__ == "__main__":

    cam = Camera(path="../../2/origin (2).avi", record=RECORD) # Setup camera with path 

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
       
