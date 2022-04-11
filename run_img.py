import os
import random
import time
import cv2
#import wx
import darknet
import numpy as np

from pypylon import pylon
from time import sleep
from time import localtime, strftime
from multiprocessing import Process, Queue
from relay import Relay


class Predict:
    def __init__(self, line, queue):
        """ 그래픽카드 할당 """
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.img = None  # 원본 이미지
        camera_num = None  # 카메라
        self.queue = queue
        self.window_name = line + '-line'
        self.line = line

        if line == 'A':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            """ 카메라 기본설정 """
            self.camera_setting = 'camera_setting_26.pfs'
            camera_num = 0
            """ 리젝트 및 사진저장 기준 설정 """
            self.Reject_limit = 78
            self.save_img_limit = 75
            
        elif line == 'B':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.camera_setting = 'camera_setting_26.pfs'
            camera_num = 1
            self.Reject_limit = 78
            self.save_img_limit = 75

        """ 이미지 저장 기본경로 """
        self.img_save_path = '/home/nongshim/바탕화면/Reject_imgs'

        """ YOLO 모델 활성화 """
        self.thresh = .5  #.86
        self.hier_thresh = .99  #.99
        self.nms = .2
        self.cfg = "./data/yolov4.cfg"
        self.weights = "./backup/2/yolov4_40000.weights" ##/2/yolov4_40000.weights"
        self.data_file = "./data/obj.data"

        self.load_network()

        """ 카메라 활성 """
        self.load_camera(camera_num=camera_num)

        """ 본 검사 프로그램 가동 """
        self.Run()


    def load_camera(self, camera_num):
        """ 카메라 설정 """
        maxCamerasToUse = 1
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
        for i, self.cam in enumerate(self.cameras):
            self.cam.Attach(tlFactory.CreateDevice(devices[camera_num]))
        self.cameras.Open()
        pylon.FeaturePersistence.Load(self.camera_setting, self.cam.GetNodeMap(), True)
        self.cameras.Close()
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


    def get_img(self):
        """ 이미지를 불러들여 self.img 안에 넣기 """
        try:
            self.grabResult = self.cameras.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(self.grabResult)
            image_raw = image_raw.GetArray()
            image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            self.img = image
            self.grabResult.Release()
        except:
            pass


    def load_network(self):
        """ 네트워크 모델 생성 하여 모델, 클래스이름, 클래스별 색상을 받기 """
        random.seed(3)  # deterministic bbox colors
        self.network, self.class_names, self.class_colors = darknet.load_network(config_file=self.cfg,
                                                                                 data_file=self.data_file,
                                                                                 weights=self.weights,
                                                                                 batch_size=1)


    def predict(self):
        """ 이미지 내 사물 디텍팅 하기 """
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        darknet_image = darknet.make_image(self.width, self.height, 3)

        image_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, self.thresh, self.nms)
        darknet.free_image(darknet_image)

        return detections


    def show_img(self):
        """ FPS를 구하여 윈도우 창에 표시하고 영상 보여주기 """
        now = time.time()
        sec = now - self.start
        fps = self.num / sec
        str = "FPS : %0.01f" % fps
        img = self.img.copy()
        cv2.putText(img, str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow(self.window_name, img)
        if self.line == 'B':
            cv2.moveWindow(self.window_name, 800, 0,)
        else :
            cv2.moveWindow(self.window_name, 0, 0,)
        

        self.num += 1


    def make_dir(self):
        """ 불량 이미지 저장할 폴더 생성하고 경로로 설정하기 """
        dir_path = self.img_save_path

        if os.path.exists(
                dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))):
            dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(
                strftime("%H", localtime()))
        else:
            if os.path.exists(dir_path + "/" + str(strftime("%Y-%m-%d", localtime()))):
                try:
                    os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime())))
                except:
                    pass
                dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))
            else:
                try:
                    os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())))
                    os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime())))
                except:
                    pass
                dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))
            print("\nThe New Folder For saving Rejected image is Maked...\n")

        return dirname_reject

    def sv_img(self, detections) :
        """ 이미지 저장 """
        dirname_reject = self.make_dir()
        if detections != []:
            for label, confidence, bbox in detections:
                if label[2] == 'R' and float(confidence) > self.save_img_limit :
                    name1 = str(label) + "-" + str(strftime("%Y-%m-%d-%H-%M-%S", localtime()))
                    name2 = str(self.window_name) + ".jpg"
                    name_orig = str('[' + str(confidence) + ']') + name1 + name2
                    #print(os.path.join(dirname_reject, name_orig))
                    cv2.imwrite(os.path.join(dirname_reject, name_orig), self.img)
                    #name_orig2 = name_orig.split(".j")[:-1][0] + "_mark.jpg"
                    #cv2.imwrite(os.path.join(dirname_reject, name_orig2), maked_img)

                    #""" yolomark txt 파일 저장 """
                    #file_name = os.path.join(dirname_reject, name_orig).split(".j")[:-1][0] + ".txt"
                    #with open(file_name, "w") as f:
                    #    for label, confidence, bbox in detections:
                    #        x, y, w, h = convert2relative(image, bbox)
                    #        if label[2] == 'R' :
                    #            label = class_names.index(label)
                    #            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} \n".format(label, x, y, w, h))


    def decide_ng(self, detections):
        """ 불량이 감지되고 일정 확률 이상일 시 error_data를 전송하기 """
        if detections != []:
            for label, confidence, bbox in detections:
                if label[2] == 'R' and self.Reject_limit < float(confidence):
                    left, top, right, bottom = darknet.bbox2points(bbox)
                    print('\n', label,confidence,'>>') 
                    error_data = [self.window_name, label[0], int((bottom*0.9+top*1.1)/2)]
                    self.put_queue(error_data)

                    self.draw_box(left, top, right, bottom)

                    #print(np.array(detections)[:, 0:1].tolist(), int((bottom*0.9+top*1.1)/2))


    def draw_box(self, left, top, right, bottom):
        """ 불량이 감지 된 위치에 박스 칠하기 """
        left = int(left / self.width * self.img.shape[1])
        right = int(right / self.width * self.img.shape[1])
        top = int(top / self.height * self.img.shape[0])
        bottom = int(bottom / self.height * self.img.shape[0])

        cv2.rectangle(self.img, (left, top), (right, bottom), [0, 0, 255], 3)


    def put_queue(self, error_data):
        """ 1차 프로세스로 정보 전달 """
        self.queue.put(error_data)


    def Run(self):
        self.make_dir()
        quit = 'on'
        self.start = time.time()
        self.num = 0
        while True:
            try:
                quit = self.queue.get(timeout=0.001)
                print(quit)
            except:
                pass

            """전역변수 self.img에 새로운 영상을 받기"""
            self.get_img()

            """ 예측하기 """
            detections = self.predict()

            """ 사진 저장하기 (save_img 기준)"""
            self.sv_img(detections)

            """ 불량인지 판단하기 """
            self.decide_ng(detections)

            """ 화면 출력하기 """
            self.show_img()

            if self.num > 1000:
                self.num = 0
                self.start = time.time()

            if cv2.waitKey(1) & 0xFF == 27:
                self.queue.put(None)
                break

            if quit == 'off':
                break

        self.cameras.StopGrabbing()
        cv2.destroyAllWindows()


class Reject:
    def __init__(self, queue):
        self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
        self.relay.state(0, False)
        self.relay.h.close()
        self.relay_runtime = 0.1
        self.relay_A = 'off'
        self.relay_B = 'off'
        self.relay_A_time = time.time()
        self.relay_B_time = time.time()
        self.queue = queue

        self.resent_height_A = 1000
        self.resent_height_B = 1000
        self.Time_out = 0.001

        """ 제품 한개가 지나가는데 필요한 시간 - 제품의 위치에 따라 가변되는 시간"""
        self.reject_need_time_A = int(0.135 / self.Time_out) #0.11  <- 실제B호기임  0.125
        #비호기 현재 제품 지나가는 시간 0.125초로 설정
        self.reject_need_time_B = int(0.125 / self.Time_out) #0.11  <- 실제A호기임  0.16

        """ 리젝트 까지 필요한 대기(컨베이어 거리) 시간 - 무조건 대기하는 시간 """
        self.standby_time_A = int(0.001 / self.Time_out)  #0.02 <- 실제B호기임
        self.standby_time_B = int(0.001 / self.Time_out)  #0.04

        self.T_A = [0] * (self.reject_need_time_A + self.standby_time_A)
        self.T_B = [0] * (self.reject_need_time_B + self.standby_time_B)

        print(self.T_A)
        

        self.get_Queue()


    def get_Queue(self):
        """ Queue를 받아서 릴레이 가동을 위한 본 프로그램 """
        initial_A = 0
        initial_B = 0
        while True:
            answer = [[0]]
            ng_question_A = 'ok'
            ng_question_B = 'ok'

            time.sleep(self.Time_out)
            #answer = self.queue.get(timeout=self.Time_out)
                
            for i in range(self.queue.qsize()):
                answer_ = self.queue.get()
                if i == 0:
                    answer = answer_
                    
            #print('answer : {}...........'.format(answer))
                
            if answer is not None:
                if answer[0][0] == 'A':  #나중에 A로 변경해야함
                    #print('A : ', self.resent_height_A, int(answer[2]))
                    if self.relay_A == 'off' and self.resent_height_A - int(answer[2]) > 50:
                        delay = self.cal_delay(answer, 'A')
                        self.T_A = self.time_traveler('A', delay, 1)
                        print('A : Reject signal put...', self.resent_height_B - int(answer[2]))
                        print("*"*20)
                        ng_question_A = 'ng'
                    self.resent_height_A = int(answer[2])
                    initial_A = 0

                elif answer[0][0] == 'B': #나중에 B로 변경해야함
                    #print('B : ', self.resent_height_B, int(answer[2]))
                    if self.relay_B == 'off' and self.resent_height_B - int(answer[2]) > 50:
                        delay = self.cal_delay(answer, 'B')
                        self.T_B = self.time_traveler('B', delay, 1)
                        print('B : Reject signal put...', self.resent_height_B - int(answer[2]))
                        print("*"*20,)
                        ng_question_B = 'ng'
                    self.resent_height_B = int(answer[2])
                    initial_B = 0



            """ 가동된 릴레이가 있다면 일정 시간 확인 후 릴레이 끄기 """
            self.relay_off_cal_time()

            """ 아무것도 감지가 안되었을 시 타임 테이블 한칸씩 이동하고 첫칸에 0 삽입 """
            if ng_question_A == 'ok':
                self.T_A = self.time_traveler('A', 0, 0)
                initial_A += 1
            if ng_question_B == 'ok':
                self.T_B = self.time_traveler('B', 0, 0)
                initial_B += 1

            """ 2000프레임 동안 리젝트가 감지가 안될 시 감지 높이 초기화 """
            if initial_A == 100:
                initial_A = 0
                self.resent_height_A = 1000
            if initial_B == 100:
                initial_B = 0
                self.resent_height_B = 1000

            """ 마지막 시간이 되면 릴레이 가동 """
            if self.T_A[len(self.T_A) - 1] == 1:
                self.state(2)
            if self.T_B[len(self.T_B) - 1] == 1:
                self.state(1)

            if answer is None:
                break

    def cal_delay(self, answer, line):
        x = 0
        location = answer[1]
        height = answer[2]

        """ 최소 및 최대 높이 고정 """
        if height <= 20:
            height = 20
        if height >= 370 :
            height = 370

        if line == 'A':
            if location == 'U':
                x = int((height/370)*(self.reject_need_time_A/2) + self.reject_need_time_A/2)
            elif location == 'D':
                x = int((height/370)*(self.reject_need_time_A/2))

        if line == 'B':
            if location == 'U':
                x = int((height/370)*(self.reject_need_time_A/2) + self.reject_need_time_A/2)
            elif location == 'D':
                x = int((height/370)*(self.reject_need_time_A/2))

        print("Delay time : ", int(x), "step - ", location, height)


        return x


    def time_traveler(self, line, location, pass_):
        """ 타임테이블 한칸씩 오른쪽으로 이동시키기 """
        if line == 'A':
            temp = self.T_A
        else:
            temp = self.T_B
        temp = temp[:-1]
        temp.insert(location, pass_)

        return temp


    def relay_off_cal_time(self):
        
        """ 릴레이가 켜져있는 상태이면, 일정 시간 이상이 초과 되었을 시 릴레이 끄기 """
        if self.relay_A == 'on':
            # print(time.time() - self.relay_A_time)
            if time.time() - self.relay_A_time > self.relay_runtime:
                self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
                self.relay.state(1, False)
                self.relay_A = 'off'
                self.relay.h.close()

        if self.relay_B == 'on':
            # print(time.time() - self.relay_A_time)
            if time.time() - self.relay_B_time > self.relay_runtime:
                self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
                self.relay.state(2, False)
                self.relay_B = 'off'
                self.relay.h.close()


    def state(self, i):

        """ i에 해당되는 릴레이가 꺼져 있으면 가동하고 현재의 시간을 저장 """
        if i == 1 and self.relay_A == 'off':
            self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
            self.relay.state(1, True)
            self.relay_A = 'on'
            self.relay_A_time = time.time()
            self.relay.h.close()

        elif i == 2 and self.relay_B == 'off':
            self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
            self.relay.state(2, True)
            self.relay_B = 'on'
            self.relay_B_time = time.time()
            self.relay.h.close()


def main():
    """ 멀티프로세스간 통신을 위한 Queue를 선언 """
    taskqueue1 = Queue()  # for A line
    taskqueue2 = Queue()  # for B line
    taskqueueR = Queue()  # for Reject

    """ 멀티프로세싱 선언 """
    LINE_1 = Process(target=Predict, args=('B', taskqueue1,))
    LINE_2 = Process(target=Predict, args=('A', taskqueue2,))
    R = Process(target=Reject, args=(taskqueueR,))

    """ 멀티프로세싱 가동 """
    LINE_1.start()
    LINE_2.start()
    R.start()

    while True:
        answer1 = ''
        answer2 = ''

        """ A라인에서 불량 감지 데이터 받기 """
        for i in range(taskqueue1.qsize()):
            answer1 = taskqueue1.get()
            if i == 0:
                print('A : ', taskqueue1.qsize())
                if answer1 is not None:
                    taskqueueR.put(answer1)

        """ B라인에서 불량 감지 데이터 받기 """
        for i in range(taskqueue2.qsize()):
            answer2 = taskqueue2.get()
            if i == 0:
                print('B : ', taskqueue2.qsize())
                if answer2 is not None:
                    taskqueueR.put(answer2)

        """ 프로그램 종료 메세지 전달 """
        if answer1 is None:
            taskqueue2.put('off')
            taskqueueR.put(None)
            break

        if answer2 is None:
            taskqueue1.put('off')
            taskqueueR.put(None)
            break


if __name__ == "__main__":
    main()
