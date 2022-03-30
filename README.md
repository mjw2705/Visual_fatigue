# Visual fatigue analysis

## Description

눈 감은 시간, 눈 깜빡임 횟수, 동공 크기 변화율, 3가지 동공 움직임 특징을 기반으로 시각피로 측정

- 3가지 특징
    - 눈 감은 시간 :  0인 프레임 개수 count
    - 눈 깜빡임 횟수 : 전 프레임이 0이 아니고 현재 프레임이 0인 경우 count
    - 동공 크기 변화율 : fft를 사용해 zero-crossing point 검출, 변화 속도 계산
    

- 동공 크기 값 보간 
  - 평균1/3이하 값 0으로  → 일반 값 두개 사이 0프레임 하나 제거 → 0프레임 두개 사이에 일반값 0으로
  

### recording code
[눈 영상 취득 녹화 GUI code](https://github.com/mjw2705/Visual_fatigue/tree/main/recording_code)

### pupil_size_detection code
[동공 크기 추출 GUI](https://github.com/mjw2705/Visual_fatigue/tree/main/pupil_size_detection_code)

### analysis code
[추출된 동공 크기 분석](https://github.com/mjw2705/Visual_fatigue/tree/main/analysising_code)


## Demo
![demo](demo_video.gif)