# B2I-GAN: Anomaly detection from imaged ECG using GAN

 >ECG(electrocardiogram) 데이터는 심장 및 혈관 장애 분석에 많이 쓰이는 데이터이다. <br>이를 이용한 기존의 연구들은 1차원 상태인 원본 데이터를 사용해왔다. 
 <br>우리는 STFT(Short-time Fourier transform)를 이용해 데이터를 2차원으로 확장시켜 시간에 따른 주파수 분포를 얻고,  
 이렇게 얻은 주파수 영역에 대한 특징을 GAN에 학습시켜 어느 부분이 비정상인지 사용자에게 보여주는 새로운 방식의 심부전 탐지 알고리즘을 연구한다. 

---

## 구성원
### Professor
- 박경문

### Student
- 2018110654 이해님
- 2016104106 김민석

---

## 연구 배경
고혈압 등 성인병 발생 비율이 증가함에 따라 심장 질환에 시달리는 사람들도 많아지고 있다.   
현재 심장 질환을 진단하기 위한 딥러닝 연구들이 진행되고 있지만, 정확도가 일정 수준을 넘지 못하고 있다.   
이는 1차원의 시계열 데이터에서 추출할 수 있는 특징에 한계가 있어 나타나는 현상일 수 있다고 보고, 원래 데이터에 STFT를 적용해 주파수 영역의 특징을 학습에 사용 하고자 한다.  

## 연구 목표
본 연구에서는 ECG데이터의 새로운 특징을 추출하기 위해 시계열 데이터(1차원)를 STFT 변환을 통해 2차원 데이터를 생성한다.  
그리고 이를 분석하는 모델을 만들어 높은 정확성을 가지는 심부전 탐지를 하고자 한다.   
정상 데이터로 이용해 학습을 진행해 어느부분이 잘못되었는지 보여주는 것이 최종 목표이다.   

## 전체 흐름

### 1. 데이터 전처리
<img src="https://user-images.githubusercontent.com/57976156/122515177-07156f00-d048-11eb-92bb-f9c588fcfa5c.PNG" width="500" height="200">
데이터를 STFT 변환을 하여 저장한다.

### 2. Train
<img src="https://user-images.githubusercontent.com/57976156/122536990-14d6ee80-d060-11eb-946e-7e18341436c4.png" width="500" height="400">
AE 기반의 Generator와 Discriminator를 통해 GAN 모델을 구성해 학습을 시킨다.   
이 때 입력 데이터는 정상 데이터만으로 학습을 진행한다. 데이터를 모델에 입력하면 정상 데이터에 근접한 이미지를 출력해 준다.  
이 과정을 반복하면서 generator는 정상 데이터의 특징을 더 잘 학습하게 되기 때문에
비정상 데이터가 입력으로 들어오더라도 정상에 가까운 fake 이미지를 출력할 수 있게 된다. 

### 3. Test
![image](https://user-images.githubusercontent.com/50744156/122553327-8029bc00-d072-11eb-9cc1-8d4a3240ec8d.png)
모델에 입력된 real 데이터와 모델을 거쳐 생성된 정상 데이터의 특징을 가지는 fake 데이터가 생성된다.  
real 데이터와 fake 데이터를 비교하여 입력 데이터가 정상 데이터에 비해 얼마나 비정상 인지,   
그리고 비정상이 발생했다면 정확히 어느 부분에서 발생했는 지를 시각적으로 보여준다.  

## Dataset
해당 연구에 사용한 데이터는 MIT-BIH Arrhythmia Database이다.   
이 데이터베이스는 다양한 종류의 심부전증을 나타내는 데이터를 포함하고 있으며 ECG 딥러닝 분야의 많은 연구에서 사용되어 왔다.   
내부 데이터는 초당 360Hz로 sampling된 신호로, 우리는 이 전체 샘플 중 가장 특징을 잘 나타내는 II-lead에 대해   
정상 신호(N)  SVE (S), VEBs (V)를 앓고 있는 환자의 ECG 신호를 추출해 GAN의 입력 데이터로 사용하였다. 다운로드 링크는 아래와 같다.  
>https://www.dropbox.com/sh/b17k2pb83obbrkn/AABF9mUNVdaYwce9fnwXsg1ta/ano0?dl=0&subfolder_nav_tracking=1  

다운받은 데이터는 /experiments/ecg/dataset/preprocessed/ano0 에 넣어준다.  
## Require
- Python 3

### Packages
- PyTorch (1.0.0)
- scikit-learn (0.20.0)
- biosppy (0.6.1) # For data preprocess
- tqdm (4.28.1)
- matplotlib (3.0.2)


## 사용법

### 데이터 전처리
원본 데이터를 STFT로 변경하여 저장해야 한다. /experiments/ecg/dataset/preprocessed에 존재하는 change.py, change2.py를 하위 디렉토리 /ano0에 넣은 후 순서대로 실행하면 된다.

### 실행
- train/test를 정하기 위해 run_ecg.sh 파일을 수정해야 한다.
    ![캡처](https://user-images.githubusercontent.com/57976156/122516269-86577280-d049-11eb-97c2-aae0b311c19a.PNG){: width="50" height="50"}
test 값이 0이면 train, 1이면 test이다.
- 실행의 명령어는 `run_ecg.sh`파일이 존재하는 디렉토리에서 다음과 같이 입력한다.<br>
`/bin/bash run_ecg.sh`

---
## 결과

![image](https://user-images.githubusercontent.com/50744156/122551614-2f18c880-d070-11eb-8e24-266d8af2aba8.png)
<br>

1차원 signal 데이터를 그대로 사용하는 beatgan과 B2I-GAN의 anomaly scoe 비교 결과는 위와 같다.  
O_N은 정상 데이터에 대한 anomaly score 분포이고 A는 비정상 데이터들(Q, S, V, F) 전체에 대한 anomaly score 분포이다.   
Anomaly score가 높을수록 정상 데이터와 다른 특징을 많이 가진다는 의미인데, BeatGAN에서 A는 대부분이 0.1이하로 정상데이터와 겹치는 부분이 많다.    
반면 B2I-GAN에서 A의 분포는 대부분이 0.1이상으로 높게 나타나는 것을 확인할 수 있다.   
이는 우리 모델이 비정상 데이터의 anomaly score를 도출하는 데 있어서 BeatGAN보다 뛰어난 성능을 보이고 있음을 나타낸다.   
또한 전체적으로 O_N과 겹치지 않는 A의 면적이 B2I-GAN에서 눈에 띄게 넓은 것을 보아 우리의 모델이 
B2I-GAN에 비해 비정상 탐지를 더 효과적으로 수행하고 있음을 알 수 있다.
  
   
<br>

|Model|AUC |AP |
|---|---|---|
|AE|0.8944|0.8415|
|AnoGAN|0.8642|0.8035|
|Ganomaly|0.9083|0.8701|
|BeatGAN|0.9447|0.9143|
|B2I-GAN|0.9460|0.9058|

<br>
  
1차원의 signal 데이터를 사용하는 다른 모델들과의 AUC 비교결과 우리 모델의 AUC가 가장 높게 나타났다.  
이를 통해 주파수 영역의 특징은 anomaly detection에 중요한 의미를 가진다는 것을 밝혀냈고 우리의 연구도 성공적이었다고 할 수 있다.


