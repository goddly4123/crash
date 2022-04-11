import hid
from _thread import *
from threading import Timer
import time
from relay import Relay
import socket

def threaded(client_socket, addr):
    print('Connected by :', addr[0], ':', addr[1])
    global data_decodeed
    # 클라이언트가 접속을 끊을 때 까지 반복합니다.
    while True:
        try:
            # 데이터가 수신되면 클라이언트에 다시 전송합니다.(에코)
            data = client_socket.recv(512)
            if not data:
                print('Disconnected by ' + addr[0])
                break
            #if str(data.decode()) != '' :
            #if '1R*' in str(data.decode()) or '2R*' in str(data.decode()) : # 통신이 끝나서 모든 데이터 받았음을 의미함. *   if str(data.decode())[-1] == '*'
            #    print(str(data.decode()))
            data_decodeed = str(data.decode())
            print('Received from ' + addr[0], ':', addr[1], data_decodeed)

            client_socket.send(data)
        except ConnectionResetError as e:
            print('Disconnected by ' + addr[0], addr[1], data_decodeed)
            break
    client_socket.close()


def run_socket():
    while True:
        client_socket, addr = server_socket.accept()
        start_new_thread(threaded, (client_socket, addr))

'''
data_decodeed 상태 확인 후 리젝트 실시
'''
def multi_relay(count_1, count_2):
    global relay_1
    global relay_2
    global data
    global data_decodeed
    if '1R*' == data_decodeed or '2R*' == data_decodeed :
        if '1' in data_decodeed and 'R' in data_decodeed and count_1 == 0 :
            relay.state(1, on=True)
            relay_1, data_decodeed = 'off_wait', ''
            print('Relay 1 : ', relay_1)
        if count_1 > 700 :
            relay.state(1, on=False)
            relay_1, data_decodeed = 'off', ''
            print('Relay 1 : ', relay_1)
        if '2' in data_decodeed and 'R' in data_decodeed and count_2 == 0 :
            relay.state(2, on=True)
            relay_2, data_decodeed = 'off_wait', ''
            print('Relay 2 : ', relay_2)
        if count_2 > 700:
            relay.state(2, on=False)
            relay_2, data_decodeed = 'off', ''
            print('Relay 2 : ', relay_2)    

if __name__ == "__main__":

    """ 릴레이 라이브러리 불러오기 """
    relay = Relay(idVendor=0x16c0, idProduct=0x05df)
    
    """ 변수설정 """
    relay_1, relay_2 = 'off', 'off'
    time_cycle = 0.0001
    data_decodeed = ''
    count_1, count_2 = 0, 0
    
    ''' 소켓통신 무한 대기상태 만들기 '''
    HOST, PORT = '127.0.0.1', 9999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print('server start')
    
    ''' 소켓통신 가동하기 '''
    start_new_thread(run_socket, ())

    ''' 본 프로그램 가동  '''
    start = time.time()
    while True:
        t = Timer(time_cycle, multi_relay, args=(count_1,count_2,)) # 0.0001초 마다 while 문 가동하면서 리젝트 작동시킴
        t.start()
        t.join()
        '''
        리젝트가 열리면 off -> off_wait로 변함, off_wait가 되면 카운트를 올려서 700이 되면 자동 릴레이 닫기
        '''
        if relay_1 == 'off_wait' : count_1 += 1
        if relay_2 == 'off_wait' : count_2 += 1
        if relay_1 == 'off' : count_1 = 0
        if relay_2 == 'off' : count_2 = 0
