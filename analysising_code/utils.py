import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack, signal


# 고주파만 날림
def get_filtered_data(in_data, filter_value=0.004):
    sig_fft = fftpack.fft(in_data)
    sample_freq = fftpack.fftfreq(in_data.size)
    high_freq_fft = sig_fft.copy()

    high_freq_fft[np.abs(sample_freq) > filter_value] = 0
    filtered_data = fftpack.ifft(high_freq_fft)

    return filtered_data

# 저주파, 고주파 날림
def get_filtered_data1(in_data, high_filter_value, low_filter_value):
    sig_fft = fftpack.fft(in_data)
    sample_freq = fftpack.fftfreq(in_data.size)
    high_freq_fft = sig_fft.copy()

    low_value1 = np.max(high_freq_fft)
    high_freq_fft[np.abs(sample_freq) > high_filter_value] = 0
    high_freq_fft[np.abs(sample_freq) < low_filter_value] = 0

    low_value2 = np.max(high_freq_fft)
    filtered_data = fftpack.ifft(high_freq_fft)

    return filtered_data, low_value1, low_value2



# 평균1/3이하인 값들 0으로
def thres_zero(y):
    countzero = y.count(0)
    countminus = y.count(-1)
    minus = -1 * countminus
    num = len(y) - countzero - countminus

    avg_nonzero = (sum(y) - (-minus)) / num
    thres = int(avg_nonzero / 3)

    y = np.array(y)
    new_y = np.where(y < thres, np.where(y > 0, 0, y), y)

    return list(new_y)


# 앞/뒤 프레임 사이에 0프레임이 하나일때 앞/뒤 프레임의 평균
def interpolation_zero(y):
    for i in range(1, len(y) - 1):
        if y[i] == 0:
            if (y[i - 1] != 0) and (y[i + 1] != 0):
                y[i] = (y[i - 1] + y[i + 1]) // 2

    return y


# 앞/뒤 프레임이 0이고, 그 사이 값이 0이 아닐 때 0으로
def interpolation_value(y):
    for i in range(1, len(y) - 1):
        if y[i] != 0:
            if (y[i - 1] == 0) and (y[i + 1] == 0):
                y[i] = 0

    return y


def first_zero(y):
    if y[0] == 0:
        if y[1] != 0:
            y[0] = y[1]
    return y


def create_folder(save_path):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except OSError:
        print(f"Error: can't create directory {save_path}")


def count_blink(pupil_list, minu=None, quar=None):
    # while -1 in pupil_list:
    #     pupil_list.remove(-1)

    # 전체 영상을 minu단위로 분석
    if minu is not None:
        time = minu * 1800  # 프레임수, 동공영상은 30fps
        chunk = len(pupil_list) // time  # 입력받은 csv파일이 몇개의 분기로 나뉘는지
    # 전체 영상을 quar개로 쪼개서 분석
    if quar is not None:
        time = len(pupil_list) // quar
        chunk = quar

    pupil_frame = []
    pupil_blink = []

    if (chunk < 1):
        pupil_frame.append(-1)
        pupil_blink.append(-1)

    for i in range(chunk):
        # 눈 감은 프레임수 세기
        count_frame = pupil_list[i * time:(i + 1) * time].count(0)
        pupil_frame.append(count_frame)

        # 눈 감은 횟수 세기
        count = 0
        for idx in range(time * i, time * (i + 1)):
            if idx >= len(pupil_list):
                break
            if pupil_list[idx] == 0 and pupil_list[idx - 1] != 0:
                count += 1  # 눈 깜빡임 count
        pupil_blink.append(count)

    return pupil_frame, pupil_blink


def normalize(pupil_frames, pupil_blinks):
    pupil_frame = np.array(pupil_frames)
    pupil_blink = np.array(pupil_blinks)
    frame_norm = []
    blink_norm = []
    sum_f = np.sum(pupil_frame)
    sum_b = np.sum(pupil_blink)
    for f_value, b_value in zip(pupil_frame, pupil_blink):
        f_norm = f_value / sum_f
        b_norm = b_value / sum_b
        frame_norm.append(f_norm)
        blink_norm.append(b_norm)

    return frame_norm, blink_norm


def draw_graph(norm_value, title, y_lim, y_label, savepath, quar=None):
    if quar is None:
        period = ['0~3분', '3~6분', '6~9분', '9~12분', '12~15분', '15~18분', '18~21분', '21~24분', '24~27분', '27~30분', '30~33분']
    else:
        period = [i+1 for i in range(quar)]
    plt.rcParams["font.family"] = 'Malgun Gothic'

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    for i, x in enumerate(norm_value):
        ax.bar(period[i], x, color='b', alpha=0.5)

    plt.xticks(rotation=20)
    plt.title(f'{title}')
    plt.ylim(0, y_lim)
    plt.xlabel('구간')
    plt.ylabel(f'{y_label}')
    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    spl = title.split('.')[0]
    plt.savefig(f'{savepath}')
    plt.imshow(img)


def fft(pupil_list, minu=None, quar=None):
    global section_frames, time

    # 데이터에서 0, -1인 부분 제거
    while 0 in pupil_list:
        pupil_list.remove(0)
    while -1 in pupil_list:
        pupil_list.remove(-1)

    if minu is not None:
        time = minu * 1800
        section_frames = len(pupil_list) // time

    if quar is not None:
        time = len(pupil_list) // quar
        section_frames = quar

    y = np.array(pupil_list)

    # fft
    filtered_sig = get_filtered_data(y, filter_value=0.005)
    # filtered_sig, _, _ = get_filtered_data1(y, 0.0048, 0.0035)
    filtered_sig = filtered_sig.astype(np.float64)

    # zero-crossing point
    zero_crossings = np.where(np.diff(np.sign(np.diff(filtered_sig))))[0]
    zero_crossings = np.insert(zero_crossings, 0, 0)
    zero_crossings = np.append(zero_crossings, len(filtered_sig) - 1)

    # 변화 속도 계산
    change_rates_list = [[] for _ in range(section_frames)]
    for section in range(section_frames):
        # zero-crossing points 기준으로 원하는 위치(섹션) 가져오기
        section_zero_crossing = zero_crossings[np.where(zero_crossings <= (section + 1) * time)]
        section_zero_crossing = section_zero_crossing[np.where(section * time < section_zero_crossing)]
        # 변화 속도 계산
        for j in range(len(section_zero_crossing) - 1):
            change_rate = abs((filtered_sig[section_zero_crossing[j + 1]] - filtered_sig[section_zero_crossing[j]]) / (
                        section_zero_crossing[j + 1] - section_zero_crossing[j]))
            change_rates_list[section].append(change_rate)

    return filtered_sig, zero_crossings, section_frames, change_rates_list


def draw_fft_graph(y, filtered_sig, zero_crossings, section_frames, title, savepath, minu=None, quar=None):
    global time
    x = np.arange(0, len(y))
    if minu is not None:
        time = minu * 1800
        section_frames = len(y) // time

    if quar is not None:
        time = len(y) // quar
        section_frames = quar

    fig = plt.figure(dpi=150)

    # plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = 'Malgun Gothic'
    plt.figure(figsize=(14, 6))
    plt.plot(x, y, label='Original signal')
    plt.plot(x, filtered_sig, linewidth=2, label='Filtered signal')
    plt.plot(zero_crossings, filtered_sig[zero_crossings], marker='o', color='red', linestyle='--')
    plt.legend(loc='upper right')

    # 섹션 나눠진거 표시
    for section in range(section_frames):
        plt.axvline(x=section * time, ymin=0, ymax=1.0, color='r')
        plt.axvline(x=(section + 1) * time, ymin=0, ymax=1.0, color='r')

    plt.xlim(1800, 4200)
    plt.title(f'{title}')
    plt.xlabel('Frame')
    plt.ylabel('Pupil size')
    plt.savefig(f'{savepath}')
    plt.show()


# 2차식 추세선 그리기, 히스토그램 그래프 저장
# 추세식 : y = a*x^2 + b*x + c
def draw_trendline(data, title, y_lim, y_label, savepath, quar = None, avg = False):
    results = {}

    # 추세선
    x = np.arange(0, len(data))
    y = []
    for idx, value in enumerate(data):
        y.append(value)
    y = np.array(y)  # 10개 구간에 해당하는 특징(깜빡임 횟수)

    fit = np.polyfit(x, y, 2)
    a = fit[0]
    b = fit[1]
    c = fit[2]
    fit_equation = a * np.square(x) + b * x + c
    results['coeffs'] = fit.tolist()

    # r-squared
    p = np.poly1d(fit)

    # fit values, and mean
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['r-squared'] = ssreg / sstot
    r_squared = str(round(results['r-squared'], 3))  # 출력하기 위해 문자열로 변환
    a = str(round(results['coeffs'][0], 3))
    b = str(round(results['coeffs'][1], 3))
    c = str(round(results['coeffs'][2], 3))
    # print("R 제곱값: ", round(results['r-squared'], 3))
    # print("추세선: "+"Y="+a+"xX^2 + "+b+"xX + "+c)

    period = ['0~3분', '3~6분', '6~9분', '9~12분', '12~15분', '15~18분', '18~21분', '21~24분', '24~27분', '27~30분', '30~33분']
    plt.rcParams["font.family"] = 'Malgun Gothic'

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    for idx2, value2 in enumerate(data):
        ax.bar(period[idx2], value2, color='b', alpha=0.5)
    ax.plot(x, fit_equation, color='r', alpha=0.5, label='Polynomial fit', linewidth=3.0)
    # ax.scatter(x, y, s = 5, color = 'b', label = 'Data points')  # 추세선 예측에 사용한 좌표 그리기

    # Plotting
    plt.xticks(rotation=20)
    plt.title(f'{title}')
    plt.ylim(0, y_lim)
    plt.xlabel('구간')
    plt.ylabel(f'{y_label}')

    # 동공 크기 변화율 출력할 때 위치 조정
    if not avg:
        plt.text(3.2, 0.055, "추세선: " + r'$y = $' + a + r'$x^2 + ($' + b + r'$)x + $' + c, fontdict={'size': 12})
        plt.text(7.5, 0.05, r'$R^2 =$' + r_squared, fontdict={'size': 12})

    # 평균 동공크기 변화율 출력할 때 위치 조정
    else:
        plt.text(3.2, 0.027, "추세선: " + r'$y = $' + a + r'$x^2 + ($' + b + r'$)x + $' + c, fontdict={'size': 12})
        plt.text(7.5, 0.025, r'$R^2 =$' + r_squared, fontdict={'size': 12})


    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    spl = title.split('.')[0]
    plt.savefig(f'{savepath}')
    plt.imshow(img)
    plt.show()  # 그래프 잘 나오는지 띄우기