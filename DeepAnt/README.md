# :books: DeepAnt: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series

- 이상을 학습하는 이상 탐지 방법론들과 달리 `label이 지정되지 않은 데이터를 사용`하여 시계열의 정상적인 동작을 예측하는 데 사용되는 `데이터 분포를 캡처(capture)하고 학습`
- 모델 생성시 이상 레이블에 의존하지 않기에 `실제 시나리오에 직접 적용이 가능`

## :bulb: Two Modules of DeepAnt
1. `Time Series Predictor`
    - CNN을 사용하여 직후의 timestamp 예측
    - (window, target)과 같이 구성하고 target은 단일 timestamp 값을 의미
2. `Anomaly Detector`
    - 예측 모듈에서 예측한 값에 대해 정상과 이상을 판별하는 모듈
    - 실제값과 예측값의 차이가 계산됨
    - 유클리드 거리가 이상 점수로 사용됨 (코드 상에서는 맨해튼 거리를 이상 점수로 활용)

## :bulb: Contribution
1. Unsupervised 설정에 있어서 시계열 데이터의 포인트 이상, context 이상 및 불일치를 감지할 수 있는 `최초의 딥러닝 기반 접근 방식`
2. 제안된 파이프라인은 유연하며 다양한 사용 사례 및 도메인에 `쉽게 적용할 수 있음`
3. 단변량 시계열 뿐만 아니라 `다변량 시계열에도 적용이 가능`
4. LSTM 기반의 접근 방식들과 다르게 CNN을 기반으로 하는 DeepAnt는 `작은 데이터셋으로도 충분히 활용할 수 있음`
5. 10개의 데이터셋에 대해 15가지 방법론들과 비교해본 결과 `대부분의 결과에서 우수한 성능을 보임`

---
### :postbox: Reference
- Original paper : https://ieeexplore.ieee.org/document/8581424
- Reference link : https://github.com/bmonikraj
- Data description : https://www.kaggle.com/aturner374/eighty-years-of-canadian-climate-data 
- Original author kaggle notebook link : https://www.kaggle.com/bmonikraj/unsupervised-timeseries-anomaly-detection/notebook 
