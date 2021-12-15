import os
import csv
import numpy as np
import pandas as pd
from utils import *


"""
csv 보간하고 새로운 csv 만드는 코드
"""


def main():
    path = '../2세대 마스크 데이터'
    save_path = './renew_csv'
    create_folder(save_path)

    file_list = os.listdir(path)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]

    for i, file in enumerate(file_list_csv):
        csv_dir = os.path.join(path, file)
        csv_file = pd.read_csv(csv_dir)
        pupil_list = csv_file['pupil_size_diameter'].tolist()

        interpol_first = first_zero(pupil_list)
        interpol_zero = interpolation_zero(interpol_first)
        interpol_val = interpolation_value(interpol_zero)
        thresh_list = thres_zero(interpol_val)

        # 보간, 수정해서 csv로 뿌리기
        name = file.split('.')[0]
        with open(f'{save_path}/{name}.csv', 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'pupil_size_diameter'])
            for i, a in enumerate(thresh_list):
                writer.writerow([i, a])


if __name__ == '__main__':
    main()
