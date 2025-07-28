import cv2
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import tkinter as tk
from tkinter import messagebox

### KEY ###
KEY_LastFrame = 'z'
KEY_NextFrame = 'x'
KEY_Last50Frame = 'd'
KEY_Next50Frame = 'f'
KEY_ClearFrame = 'c'
KEY_AimBot = 'v'
KEY_Event0 = '0'
KEY_Event1 = '1'
KEY_Event2 = '2'
KEY_Event3 = '3'
KEY_Save = 's'
KEY_Quit = 'q'
KEY_Esc = 27
KEY_Up = 'i'
KEY_Left = 'j'
KEY_Down = 'k'
KEY_Right = 'l'

def confirm_exit():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    response = messagebox.askyesno("Confirm Exit", "Are you sure you want to quit?")
    root.destroy()
    return response

### Video ###
if len(sys.argv) < 2 or os.path.isdir(sys.argv[1]):
    if len(sys.argv) < 2:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(sys.argv[1])
    for i, f in enumerate(os.listdir('video')):
        if os.path.isfile(os.path.join('csv', os.path.splitext(f)[0] + '_ball.csv')):
            print(f'{i+1:3d}: {f} (labeled)')
        else:
            print(f'{i+1:3d}: {f}')
    idx = int(input('Enter the video number: '))
    # if os.path.isfile(os.path.join('csv', os.path.splitext(f)[0] + '_ball.csv')):
    #     TRACKNET = True
    # else:
    #     TRACKNET = False
    VIDEO_NAME = 'video/' + os.listdir('video')[idx-1]
else:
    VIDEO_NAME = sys.argv[1]


### CSV ###
parent_dir = os.path.dirname(os.path.dirname(VIDEO_NAME))
csv_dir = os.path.join(parent_dir, 'csv')
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
csv_file_name = os.path.splitext(os.path.basename(VIDEO_NAME))[0] + '_ball.csv'
CSV_NAME = os.path.join(csv_dir, csv_file_name)
COLUMNS = ['Frame', 'Visibility', 'X', 'Y']

### FRAME ### 
frame_dir = os.path.join(parent_dir, 'frame')
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

### Other Variables ###
ZOOM_SCALE = 1
ZOOM_INCREMENT = 1.05


### Label Color ###
# Event: Color
LABEL_COLOR = {0: (0, 0, 255),
               1: (0, 255, 255),
               2: (0, 255, 0),
               3: (255, 0, 0)}
LABEL_SIZE = 3
AIMBOT_RANGE = 50
TEXT_SIZE = 0.8
TEXT_COLOR = (0, 255, 255)
sift = cv2.SIFT_create()
frames = []

def tracknet():
    # 執行 target.py 並取得輸出
    result = subprocess.run(["python", "predict.py", "--video_name", VIDEO_NAME, "--load_weight", "weights/TrackNet10_30_cut.tar"], capture_output=True, text=True)
    print(result.stdout)  # 打印 target.py 的輸出

def aimbot(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("Error in cvtColor")
        return -1, -1

    # Detect keypoints using ORB
    keypoints, _ = sift.detectAndCompute(gray, None)
    kp_x, kp_y = -1, -1

    if keypoints:
        # Sort keypoints by their response (significance) and take the strongest one
        most_significant_kp = max(keypoints, key=lambda kp: kp.response)
        kp_x, kp_y = int(most_significant_kp.pt[0]), int(most_significant_kp.pt[1])

        # Display keypoints on the cropped image
        cropped_with_kp = cv2.drawKeypoints(img, [most_significant_kp], None, color=(0, 255, 0), flags=0)

        # Show the cropped image with the most significant keypoint highlighted
        cropped_with_kp = cv2.resize(cropped_with_kp, (cropped_with_kp.shape[1] * 2, cropped_with_kp.shape[0] * 2))
        # cv2.imshow('Cropped Image with Keypoint', cropped_with_kp)
    return kp_x, kp_y
        

def empty_row(frame_idx : int, fps : float):
    return {'Frame': frame_idx, 'Visibility': 0, 'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'Event': 0, 'Timestamp': frame_idx/fps}

def dict_save_to_dataframe(df):
    pd_df = pd.DataFrame.from_dict(df, orient='index', columns=COLUMNS)
    pd_df.to_csv(CSV_NAME, encoding = 'utf-8',index = False)

def on_Mouse(event, x, y, flags, param):
    # print(x, y)
    global ZOOM_SCALE, img, scale_flag, scale_X, scale_Y, half_W, half_H
    # if event == cv2.EVENT_MOUSEMOVE:
    # img = cv2.rectangle(img, (x - AIMBOT_RANGE, y - AIMBOT_RANGE), (x + AIMBOT_RANGE, y + AIMBOT_RANGE), (0, 255, 0), 1)

    if scale_flag:
        x = int(x + scale_X - half_W)
        y = int(y + scale_Y - half_H)
        

    if event == cv2.EVENT_RBUTTONDOWN:
        return
        # x1 = max(0, x - AIMBOT_RANGE)
        # y1 = max(0, y - AIMBOT_RANGE)
        # x2 = min(width, x + AIMBOT_RANGE)
        # y2 = min(height, y + AIMBOT_RANGE)
        
        # cropped_img = img[y1:y2, x1:x2]
        
        # # if scale_flag:
        # #     x = float(x + scale_X - half_W)
        # #     y = float(y + scale_Y - half_H)
        # df[current_frame_idx]['Visibility'] = 1
        # kp_x, kp_y = aimbot(cropped_img)
        # if kp_x != -1 and kp_y != -1:
        #     df[current_frame_idx]['X'] = x1 + kp_x
        #     df[current_frame_idx]['Y'] = y1 + kp_y
        # else:
        #     df[current_frame_idx]['X'] = x
        #     df[current_frame_idx]['Y'] = y
        # print(f"X: {df[current_frame_idx]['X']}, Y: {df[current_frame_idx]['Y']}")
        if df[current_frame_idx-1]['Visibility'] and df[current_frame_idx-2]['Visibility'] and df[current_frame_idx-3]['Visibility'] and df[current_frame_idx-4]['Visibility']:
            # use polynomial interpolation to use previous 3 frames to predict next frame
            t = np.array([1, 2, 3, 4])
            x = np.array([df[current_frame_idx-4]['X'], df[current_frame_idx-3]['X'], df[current_frame_idx-2]['X'], df[current_frame_idx-1]['X']])
            y = np.array([df[current_frame_idx-4]['Y'], df[current_frame_idx-3]['Y'], df[current_frame_idx-2]['Y'], df[current_frame_idx-1]['Y']])

            # 使用 np.polyfit 進行二次多項式擬合
            # x 和 y 分別對時間 t 進行擬合
            coeff_x = np.polyfit(t, x, 2)

            coeff_y = np.polyfit(t, y, 2)

            # 擬合的係數是 (a, b, c)，對應於 y = at^2 + bt + c
            a_x, b_x, c_x = coeff_x
            a_y, b_y, c_y = coeff_y

            # 預測第四個點的時間 t4
            t5 = 5

            # 根據擬合的多項式公式預測 x4 和 y4
            x5 = a_x * t5**2 + b_x * t5 + c_x
            y5 = a_y * t5**2 + b_y * t5 + c_y

            df[current_frame_idx]['Visibility'] = 1
            df[current_frame_idx]['X'] = x5
            df[current_frame_idx]['Y'] = y5

    

        elif df[current_frame_idx-1]['Visibility'] and df[current_frame_idx-2]['Visibility'] and df[current_frame_idx-3]['Visibility']:
            # use polynomial interpolation to use previous 3 frames to predict next frame
            t = np.array([1, 2, 3])  # 時間序列 (假設每個 frame 之間的時間間隔相同)
            x = np.array([df[current_frame_idx-3]['X'], df[current_frame_idx-2]['X'], df[current_frame_idx-1]['X']])
            y = np.array([df[current_frame_idx-3]['Y'], df[current_frame_idx-2]['Y'], df[current_frame_idx-1]['Y']])

            # 使用 np.polyfit 進行二次多項式擬合
            # x 和 y 分別對時間 t 進行擬合
            coeff_x = np.polyfit(t, x, 2)  # x(t) 的二次擬合
            coeff_y = np.polyfit(t, y, 2)  # y(t) 的二次擬合

            # 擬合的係數是 (a, b, c)，對應於 y = at^2 + bt + c
            a_x, b_x, c_x = coeff_x
            a_y, b_y, c_y = coeff_y

            # 預測第四個點的時間 t4
            t4 = 4  # 例如預測第四個 frame 時的座標

            # 根據擬合的多項式公式預測 x4 和 y4
            x4 = a_x * t4**2 + b_x * t4 + c_x
            y4 = a_y * t4**2 + b_y * t4 + c_y

            df[current_frame_idx]['Visibility'] = 1
            df[current_frame_idx]['X'] = x4
            df[current_frame_idx]['Y'] = y4


        elif df[current_frame_idx-1]['Visibility'] and df[current_frame_idx-2]['Visibility']:
            x1 = int(df[current_frame_idx-2]['X'])
            y1 = int(df[current_frame_idx-2]['Y'])
            x2 = int(df[current_frame_idx-1]['X'])
            y2 = int(df[current_frame_idx-1]['Y'])
            df[current_frame_idx]['Visibility'] = 1
            df[current_frame_idx]['X'] = x2 + (x2 - x1)
            df[current_frame_idx]['Y'] = y2 + (y2 - y1)
            print(f"X: {df[current_frame_idx]['X']}, Y: {df[current_frame_idx]['Y']}")


    elif event == cv2.EVENT_LBUTTONDOWN:
        
        df[current_frame_idx]['Visibility'] = 1
        df[current_frame_idx]['X'] = x
        df[current_frame_idx]['Y'] = y
        # df[current_frame_idx]['Event'] = 1       
        print(f"X: {df[current_frame_idx]['X']}, Y: {df[current_frame_idx]['Y']}")

    if event == cv2.EVENT_MOUSEWHEEL:
        scale_flag = False
        if flags > 0:
            # Scroll up to zoom in
            ZOOM_SCALE = min(5.0, ZOOM_SCALE * ZOOM_INCREMENT)
        else:
            # Scroll down to zoom out
            ZOOM_SCALE = max(1.0, ZOOM_SCALE / ZOOM_INCREMENT)
        scale_X = x
        scale_Y = y
        half_W = int(width/ZOOM_SCALE//2)
        half_H = int(height/ZOOM_SCALE//2)
        if ZOOM_SCALE != 1.0:
            scale_flag = True

        

def update_frame(cap, current_frame_idx, previous_frame_idx, current_row):
    global scale_flag, scale_X, scale_Y, half_H, half_W
    # use frames list to get frame
    frame = frames[current_frame_idx].copy()
    # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    # ret, frame = cap.read()
    # if not ret:
    #     return None
    
    # check if frame is equal to frames[current_frame_idx]
    # if not (frame == frames[current_frame_idx]).all():
    #     print(f"Frame {current_frame_idx} is broken, save and quit. Please report it.")
    # else:
    #     print(f'Frame {current_frame_idx} same')
    
    if previous_frame_idx != current_frame_idx:
        if scale_flag:
            for i in range(current_frame_idx-1, -1, -1):
                if df[i]['Visibility']:
                    scale_X = int(df[i]['X'])
                    scale_Y = int(df[i]['Y'])
                    break

    # if Ball, Draw Color on label
    if current_row['Visibility']:
        frame = cv2.circle(frame, (int(current_row['X']), int(current_row['Y'])), LABEL_SIZE, LABEL_COLOR[0], -1)
    # frame = cv2.putText(frame, f"Event: {current_row['Event']}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 2)
    return frame

if __name__ == '__main__':
    print(f"-------------------------\n"
          f"| Last Frame:          {KEY_LastFrame}\n"
          f"| Next Frame:          {KEY_NextFrame}\n"
          f"| Last 50 Frame:       {KEY_Last50Frame}\n"
          f"| Next 50 Frame:       {KEY_Next50Frame}\n"
          f"| Clear Label & Event: {KEY_ClearFrame}\n"
          f"| Event 0:             {KEY_Event0}\n"
          f"| Event 1 (Hit):       {KEY_Event1}\n"
          f"| Event 2 (Land):      {KEY_Event2}\n"
          f"| Event 3 (Serve):     {KEY_Event3}\n"
          f"| Zoom:                Scroll\n"
          f"| Save:                {KEY_Save}\n"
          f"| Save & Quit:         {KEY_Quit}\n"
          f"| No Save & Quit:      Esc\n"
          F"| Move 1 pixel:        {KEY_Up},{KEY_Left},{KEY_Down},{KEY_Right}\n"
          f"| Quick AimBot:        {KEY_AimBot}\n"
          f"| Label w/ Aimbot:     Right Click\n"
          f"| Label w/o Aimbot:    Left Click\n"
          f"-------------------------\n")
        
    # Read Video
    cap = cv2.VideoCapture(VIDEO_NAME)
    total_frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {VIDEO_NAME}, total_frame_idx: {total_frame_idx}, width: {width}, height: {height}, fps: {fps}")
    if os.path.isfile(CSV_NAME): # Exist csv file
        df = pd.read_csv(CSV_NAME)
        assert set(COLUMNS).issubset(df.columns), f"{CSV_NAME} is missing columns from {COLUMNS}!"
        df = df.to_dict('index')
        if len(df) == 0:
            df[0] = empty_row(frame_idx=0, fps=fps)
    else: # New File
        df = {}
        df[0] = empty_row(frame_idx=0, fps=fps)

    current_frame_idx = 0 # Jump to last frame idx
    previous_frame_idx = current_frame_idx
    print(f"Start from Frame: {current_frame_idx}")

    print (f"Total Frame: {total_frame_idx}")
    print (f"Width: {width}")
    print (f"Height: {height}")
    print (f"Fps: {fps}")

    scale_flag = False
    aimbot_flag = False
    scale_X = None
    scale_Y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    img = update_frame(cap, current_frame_idx, previous_frame_idx, df[current_frame_idx])
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_Mouse, img)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    # cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN | cv2.WINDOW_KEEPRATIO, cv2.WINDOW_FULLSCREEN)

    

    while True:
        # insert a initial row
        if current_frame_idx >= len(df): # new row
            df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, fps=fps)

        # display the image and wait for a keypress
        img = update_frame(cap, current_frame_idx, previous_frame_idx, df[current_frame_idx])
        previous_frame_idx = current_frame_idx
        if img is None:
            print(f"Frame {current_frame_idx} is broken, save and quit. Please report it.")
            dict_save_to_dataframe(df)
            break

        if current_frame_idx > 0:
            lastFrame = df[current_frame_idx-1]
            lst_x, lst_y = int(lastFrame['X']), int(lastFrame['Y'])
            if lastFrame['Visibility']:
                if aimbot_flag:
                    cropped_img = img[lst_y - AIMBOT_RANGE : lst_y + AIMBOT_RANGE, lst_x - AIMBOT_RANGE : lst_x + AIMBOT_RANGE]
                    kp_x, kp_y = aimbot(cropped_img)
                    if kp_x != -1 and kp_y != -1:
                        df[current_frame_idx]['Visibility'] = 1
                        df[current_frame_idx]['X'] = lst_x - AIMBOT_RANGE + kp_x
                        df[current_frame_idx]['Y'] = lst_y - AIMBOT_RANGE + kp_y
                        print(f"X: {df[current_frame_idx]['X']}, Y: {df[current_frame_idx]['Y']}")
                    else:
                        print("aimbot fail")
                    aimbot_flag = False
                # img = cv2.rectangle(img, (lst_x - AIMBOT_RANGE, lst_y - AIMBOT_RANGE), (lst_x + AIMBOT_RANGE, lst_y + AIMBOT_RANGE), (0, 255, 0), 2)
                
        if scale_flag:
            scale_X = min(width - half_W, max(scale_X, half_W))
            scale_Y = min(height - half_H, max(scale_Y, half_H))
            zoomimg = img[scale_Y - half_H : scale_Y + half_H,
                          scale_X - half_W : scale_X + half_W]
            zoomimg = cv2.putText(zoomimg, f"Frame: {current_frame_idx:03d}/{total_frame_idx-1:03d}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 2)
            cv2.imshow('image', zoomimg)
        else:
            img = cv2.putText(img, f"Frame: {current_frame_idx:03d}/{total_frame_idx-1:03d}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 2)

            cv2.imshow('image', img)

        key = cv2.waitKeyEx(1)

        if key & 0xFFFF == ord(KEY_NextFrame):
            if current_frame_idx < total_frame_idx-1:
                current_frame_idx += 1
            if current_frame_idx >= len(df):
                df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, fps=fps)
            print(f"Current Frame: {current_frame_idx}")

        if key & 0xFFFF == ord(KEY_AimBot):
            if current_frame_idx < total_frame_idx-1:
                current_frame_idx += 1
            if current_frame_idx >= len(df):
                df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, fps=fps)
            print(f"Current Frame: {current_frame_idx}")
            aimbot_flag = True

        if key & 0xFFFF == ord(KEY_Next50Frame):
            if current_frame_idx + 50 < total_frame_idx-1:
                current_frame_idx += 50
            else:
                current_frame_idx = total_frame_idx-1
            if current_frame_idx >= len(df):
                for i in range(len(df), current_frame_idx+1):
                    df[i] = empty_row(frame_idx=i, fps=fps)
            print(f"Current Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_LastFrame):
            if current_frame_idx > 0:
                current_frame_idx -= 1
            print(f"Current Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Last50Frame):
            if current_frame_idx - 50 > 0:
                current_frame_idx -= 50
            else:
                current_frame_idx = 0
            print(f"Current Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event0):
            df[current_frame_idx]['Event'] = 0
            print(f"{KEY_Event1} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event1):
            df[current_frame_idx]['Event'] = 1
            print(f"{KEY_Event1} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event2):
            df[current_frame_idx]['Event'] = 2
            print(f"{KEY_Event2} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event3):
            df[current_frame_idx]['Event'] = 3
            print(f"{KEY_Event3} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Save):
            dict_save_to_dataframe(df)
            print(f"Save At Frame {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Quit):
            dict_save_to_dataframe(df)
            print(f"Save At Frame {current_frame_idx}")
            break

        elif key & 0xFFFF == ord(KEY_ClearFrame): # clear label of this frame
            df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, fps=fps)
            print(f"Clear Frame {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Up): # press keyboard direction up
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['Y'] -= 1

        elif key & 0xFFFF == ord(KEY_Down): # press keyboard direction down
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['Y'] += 1

        elif key & 0xFFFF == ord(KEY_Left): # press keyboard direction left
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['X'] -= 1

        elif key & 0xFFFF == ord(KEY_Right): # press keyboard direction right
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['X'] += 1

        elif key == KEY_Esc:  # Esc key pressed
            if confirm_exit():
                print("Exiting.")
                break
            else:
                print("Continuing labeling.")

