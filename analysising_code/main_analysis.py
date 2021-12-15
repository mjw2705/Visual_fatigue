# pyinstaller -F --icon=./chart_graph.ico analysising.py
import csv
import os
import pandas as pd

import sys
from utils import *


"""
눈깜빡임 횟수, 눈 감은 시간, 동공 크기 변화율 분석하는 코드
"""

def main(norm, save_img):
    global quarter, minute

    # 분석 분기 or 분 입력받기
    f = open('분석선택.txt', 'rt', encoding='utf-8-sig')
    line = f.readline().split(",")
    f.close()

    if line[0] == '분기':
        quarter = int(line[1])
        minute = None
    else:
        minute = int(line[1])
        quarter = None


    ## 콘솔창에 입력받기
    # period, num = map(str, sys.stdin.readline().split())
    # if period == '분기':
    #     quarter = int(num)
    #     minute = None
    # else:
    #     minute = int(num)
    #     quarter = None


    if save_img:
        save_img_blink = './img_blink'
        save_img_close = './img_close'
        save_img_size = './img_size'
        save_img_fft = './img_fft'
        create_folder(save_img_blink)
        create_folder(save_img_close)
        create_folder(save_img_size)
        create_folder(save_img_fft)

    path = './renew_csv'
    file_list = os.listdir(path)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]

    with open('분석 결과.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        names = []
        frames = []
        blinks = []
        sizes = []
        for i, file in enumerate(file_list_csv):
            name = file.split('.')[0]
            names.append(name)
            csv_dir = os.path.join(path, file)
            csv_file = pd.read_csv(csv_dir)
            pupil_list = csv_file['pupil_size_diameter'].tolist()

            # csv 보간
            interpol_zero = interpolation_zero(pupil_list)
            interpol_val = interpolation_value(interpol_zero)
            thresh_list = thres_zero(interpol_val)

            # 눈감은 시간, 눈깜빡임 횟수
            pupil_frames, pupil_blinks = count_blink(thresh_list, minu=minute, quar=quarter)
            # 정규화
            if norm:
                pupil_frames, pupil_blinks = normalize(pupil_frames, pupil_blinks)

            frames.append(pupil_frames)
            blinks.append(pupil_blinks)

            # 동공크기 변화율
            filter, zero_cross, section_frames, change_rates_list = fft(pupil_list, minu=minute, quar=quarter)
            # 분기 별 변화율
            change_rates = []
            for change_rate in change_rates_list:
                if change_rate:
                    average_change_rate = sum(change_rate) / len(change_rate)
                    change_rates.append(average_change_rate)
            sizes.append(change_rates)

            if save_img:
                # 눈 깜빡임 횟수, 눈 감은 시간, 동공 크기 변화율 막대 그래프 그리기
                draw_graph(pupil_frames, file, 0.3, '눈감은시간', f'{save_img_close}/{name}_눈감.png', quar=quarter)
                draw_graph(pupil_blinks, file, 0.3, '눈깜빡임횟수', f"{save_img_blink}/{name}_눈깜.png", quar=quarter)
                draw_graph(change_rates, name, 0.06, '동공크기 변화율', f'{save_img_size}/{name}_동공변화.png', quar=quarter)
                # 동공크기 변화율 그래프 그리기
                draw_fft_graph(pupil_list, filter, zero_cross, section_frames, name, f'{save_img_fft}/{name}.png')

        # 평균
        avg_frame = np.array(frames).mean(axis=0)
        avg_blink = np.array(blinks).mean(axis=0)
        avg_size = np.array(sizes).mean(axis=0)
        avg = [list(avg_frame), list(avg_blink), list(avg_size)]

        # csv 쓰기
        writer.writerow(['눈 감고있는 시간'])
        writer.writerow(names)
        for i in zip(*frames):
            writer.writerow(i)
        writer.writerow([''])
        writer.writerow(['눈 깜빡임 횟수'])
        writer.writerow(names)
        for i in zip(*blinks):
            writer.writerow(i)
        writer.writerow([''])
        writer.writerow(['동공 크기 변화율'])
        writer.writerow(names)
        for i in zip(*sizes):
            writer.writerow(i)
        writer.writerow([''])
        writer.writerow(['평균'])
        writer.writerow(['눈 감고있는 시간', '눈 깜빡임 횟수', '동공 크기 변화율'])
        for i in zip(*avg):
            writer.writerow(i)

        if save_img:
            # 전체 평균 그래프
            draw_graph(avg_frame, '눈감은 시간 평균', 0.2, '눈감은시간', f'{save_img_close}/[평균]_눈감은시간.png', quar=quarter)
            draw_graph(avg_blink, '눈깜빡임 횟수 평균', 0.2, '눈깜빡임 횟수', f'{save_img_blink}/[평균]_눈깜빡임횟수.png', quar=quarter)
            draw_graph(avg_size, '동공크기 변화율 평균', 0.05, '동공크기 변화율', f'{save_img_size}/[평균]_동공크기 변화율.png', quar=quarter)
            # 평균 추세선 그래프
            draw_trendline(avg_frame, '눈감은시간 평균', 0.15, '눈감은시간', f'{save_img_close}/[평균추세선]_눈감은시간.png', quar=quarter, avg=True)
            draw_trendline(avg_blink, '눈깜빡임횟수 평균', 0.15, '눈깜빡임 횟수', f'{save_img_blink}/[평균추세선]_눈깜빡임횟수.png', quar=quarter, avg=True)
            draw_trendline(avg_size, '동공크기 변화율 평균', 0.03, '동공크기 변화율', f'{save_img_size}/[평균추세선]_동공크기 변화율.png', quar=quarter, avg=True)


if __name__ == '__main__':
    norm = True
    save_img = True
    main(norm, save_img)