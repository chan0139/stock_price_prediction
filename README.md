# stock_price_prediction

# 개요

- 주식의 향방을 예측하는 지표 多 
- 각자의 지표는 의미를 가질 것이라는 기대
- 컴퓨터를 활용해 각자 지표 학습 / 합치면 수익 기대?
  
    -> ML/DL을 통한 주가 향방 분석

# 진행 방향

- 차트 지표
- 캔들 차트
- 수급
- 시계열
- 자연어 / 거시경제

-> KODEX 200의 상승 or 하락 예측

-> 개별주 차원으로 확장


# 차트 지표
![image](https://user-images.githubusercontent.com/33649931/171106122-80aa32fc-889b-4f82-810b-5e52372e9fb2.png)


# 캔들 차트
![image](https://user-images.githubusercontent.com/33649931/171106297-42acf99f-8ddd-4d04-8609-4528937b9b33.png)


# 수급
![image](https://user-images.githubusercontent.com/33649931/171106412-8c021324-5e13-4c95-ae1a-f85c4640081b.png)


# 시계열
![image](https://user-images.githubusercontent.com/33649931/171106487-f5a78475-a7e3-4534-a431-3393c8067409.png)

# 자연어 / 거시경제
![image](https://user-images.githubusercontent.com/33649931/171106657-0ec7d748-560e-4334-94ba-33cad9b1d8c3.png)

# SOFT VOTING
![image](https://user-images.githubusercontent.com/33649931/171109357-5a55767d-5048-4bfe-9a9b-7d626498bce4.png)

# 웹페이지

<p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171112217-ad7c4053-c34c-4a3a-b1ec-7e23178bb422.gif" alt="factorio thumbnail"/>
</p> 

![image](https://user-images.githubusercontent.com/33649931/171109751-2fce8647-ff8b-4865-8a44-64be1bb89b95.png)

![image](https://user-images.githubusercontent.com/33649931/171110820-a15ed0b5-a47b-4bb3-b5ac-66c6509d3e0b.png)

## 자연어 / 거시경제

1. 네이버 증권 주요 누스 웹크롤링
 : 2007 ~ 현재

<p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171115001-b006df60-56e4-43f7-bb25-ef3234ac452c.png" alt="factorio thumbnail"/>
</p> 


2. KoBert 모델로 자연어 처리
 : 네이버리뷰 데이터로 모델 학습

<p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171115952-c06abfef-2821-46e9-a88b-0fbf49b3baf8.png" alt="factorio thumbnail"/>
</p> 

<p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171116192-739e4dea-65fa-43dc-a251-b0f0ecc14321.png" alt="factorio thumbnail"/>
</p> 

3. 웹 크롤링 데이터 라벨 예측
 : 학습된 kobert모델로 뉴스 제목에 따라 예측하고 긍정, 부정 퍼센트에 따라 라벨링 분류
   -1 / 0 / 1 부정 / 중립 / 긍정 3가지로 분류
   날짜별로 평균 감정 지수 구하기 ( groupby)

<p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171115744-4060ea85-1acd-4b6b-a21f-f2dec835f2c2.png" alt="factorio thumbnail"/>
</p> 


4. 거시경제지표 추가
 : 나스닥, 다우, 금, 등의 지표 추가

 <p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171116597-78cf2d3c-e443-4c7f-a6c4-1f1cb3eeeb1b.png" alt="factorio thumbnail"/>
</p> 

1. LSTM 모델 적용
 : 이전 데이터를 가지고 이후의 데이터를 예측
   window size = 10 (10일 이전의 데이터를 가지고 다음날 예측)

- LSTM 모델
 <p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171116938-eea5a1f8-b993-4493-9158-8a9a34bd89ca.png" alt="factorio thumbnail"/>
</p> 

- 정확도 계산
 <p align="center" width:30px>
  <img src="https://user-images.githubusercontent.com/33649931/171116981-88b99726-80d2-4ce6-a82e-3ef13719f4f0.png" alt="factorio thumbnail"/>
</p> 
