#! /usr/bin/python3.7

import cv2

import os
import sys
import time
from socket import *
from traffic_light import *


class MainClass():
    def __init__(self, video_path = "",
                        hilens_enable = True, 
                        socket_enable = True, 
                        display_enable = True, 
                        model_enable = True, 
                        record_enable = False):
        self.__hilens_enable = hilens_enable
        self.__socket_enable = socket_enable
        self.__display_enable = display_enable
        self.__model_enable = model_enable
        self.__record_enable = record_enable
        if self.__hilens_enable:
            import hilens
            video_path=""


        if self.__socket_enable:
            self.__socket_3399 = SocketThreadGuard(socket_enable=socket_enable)
        
        if self.__display_enable:
            self.__display = hilens.Display(hilens.HDMI)
        else:
            self.__display = None
        
        model = None
        if self.__model_enable:
            # 初始化模型
            # model_path = hilens.get_model_dir() + "model-extreme-1-1.om"
            if self.__hilens_enable:
                model_path = hilens.get_model_dir() + "speed_detect.om"
            else:
                model_path = "model/convert-1360.om"
                # model_path = hilens.get_model_dir() + "convert-1360.om"

            model = hilens.Model(model_path)
            print("create model")

        self.__cam_thread = CameraThreadGuard(video_path, RECORD=self.__record_enable, HILENS=self.__hilens_enable)
        self.__pp_thread = PictureProcessorThreadGuard(model, HILENS=self.__hilens_enable, socket_enable=self.__socket_enable, display = self.__display)

    def run(self):
        
        thread_lists = [self.__cam_thread, self.__pp_thread]
        if self.__socket_enable:
            thread_lists.append(self.__socket_3399)

        for t in thread_lists:
            t.setDaemon(True)
        print("start")
        try:
            for t in thread_lists:
                t.start()

            # Wait for complete. 
            while True: 
                time.sleep(3)

            for t in thread_lists:
                t.join()

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            if self.__hilens_enable:
                hilens.terminate()


if __name__ == "__main__":
    # 系统初始化，参数要与创建技能时填写的检验值保持一致
    video_path = "E:\\aboutme\\huawei_self_driving\\videos\\traffic_light\\red_green_yellow0.mp4"
    video_path = "src_output_cut.mp4"

    hilens_enable = False
    socket_enable = False
    display_enable = False
    model_enable = False 
    if hilens_enable:
        import hilens
        hilens.init("hello")
    if hilens_enable:
        video_path = ""
    main_class = MainClass(video_path, hilens_enable, socket_enable, display_enable, model_enable)
    main_class.run()


