# 기상 데이터로 온도 회귀분석
(2020.03.06~2020.04.13)

# 프로젝트 (경진대회) 개요

- DACON 경진대회에 참가하여 제출한 프로젝트에 대한 소개
- 대학교 동기가 이미 팀빌딩된 팀을 이끌던 중, 합류 제안을 받아 참가
-----------------------------------------------------------------

[AI프렌즈 시즌1 온도 추정 경진대회 <br> 빅데이터와 AI를 이용 'My 기상청' 만들기 - MSE](https://dacon.io/competitions/official/235584/overview/)
-  대회 소개 : 저가의 센서로 관심대상의 온도를 단기간 측정하여 기상청의 관측 데이터와의 상관관계 모델을 만들고, 이후엔 생성된 모델을 통해 온도를 추정



    >데이터 설명 
    >- 대전지역에서 측정한 실내외 19곳의 센서데이터와, 주변 지역의 기상청 공공데이터를 semi-비식별화하여 제공
    >- 센서는 온도를 측정
    >- 모든 데이터는 시간 순으로 정렬 되어 있으며 10분 단위 데이터
    
    >train.csv(4752 rows × 60 columns)
    >- 30일 간의 기상청 데이터 (X00-X39) 및 센서데이터 (Y00-Y17)
    >- 이후 3일 간의 기상청 데이터 (X00-X39) 및 센서데이터 (Y18)

    |변수명|변수설명|
    |---|:---:|
    |id|         고유 번호 (시간 순서)|
    |X00, X07, X28, X31, X32|기온|
    |X01, X06, X22, X27, X29|현지기압|
    |X02, X03, X18, X24, X26|   풍속|
    |X04, X10, X21, X36, X39|   일일 누적강수량|
    |X05, X08, X09, X23, X33|   해면기압|
    |X11, X14, X16, X19, X34|   일일 누적일사량|
    |X12, X20, X30, X37, X38|   습도|
    |X13, X15, X17, X25, X35|   풍향|
    |Y00~Y17|         센서 측정 온도(NaN 432개-마지막3일 없음)|
    |Y18|         센서 측정 온도(NaN 4320개-마지막3일만 있음)|

    >test.csv(11520 rows × 41 columns)
    >- train.csv 기간 이후 80일 간의 기상청 데이터 (X00-X39)
- 예측 대상(target variable) : 80일 간의 Y18
- 채점 기준 : MSE
    ```python
    import numpy as np

    def mse_AIFrenz(y_true, y_pred):
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    diff = abs(y_true - y_pred)
    
    less_then_one = np.where(diff < 1, 0, diff)
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    
    return score
    ```
  - 최종마감날짜 이전에 [DACON 리더보드](https://dacon.io/competitions/official/235584/leaderboard/)에 여러번 예측값 제출 가능
  - 제출한 예측값을 기준으로 DACON 측에서 MSE 산출
  - 최종마감날짜 이전에 제출한 예측값 중 MSE 가 가장 낮은 예측값을 기준으로 최종 팀 순위 산출






# 분석 방향 및 과정
## 1. 데이터 탐색
- DACON 측에서 업로드한 데이터에 대한 추가적인 설명 영상을 참고

[온도 추정 대회 데이터 설명 동영상 보기](https://youtu.be/ukzaKsnKfXw)

![데이터설명1](/image/데이콘기상데이터설명.PNG) <br>
![데이터설명2](/image/데이콘기상데이터설명2.PNG)

- 합류하기 전, 팀은 단일모델을 사용하고 있었고 변수 Y00~17 처리에 대한 방안을 결정하지 못한 상태였음.<br>데이터 형태를 다시 들여다보고 방향성을 수정.

![데이터프레임](/image/경진대회데이터프레임5.PNG)

## 2. 모델링 방향 수정


#### 1. Multivariate Regressor Model 2개 사용 
    - Model 1 : X -> Y00~Y17 학습한 모델
    - Model 2 : Y00~Y17 -> Y18 학습한 모델




---
#### 2. 각 모델은 Regressor들을 앙상블
- 이전에 트리기반 Regressor 중 하나만 사용하여 학습과 예측을 했던 방식에서 여러 Regressor 를 앙상블하는 방식으로 수정.  

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor ...
from xgboost import XGBRegressor
...
```
#### 3. 최종 예측 모델 
Model 1 -> Model 2 -> Y18 예측

    - Model 1 : X_test 로 Y00~17 예측
    - Model 2 : Y00~17 예측값으로 Y18 예측


-------------------------------------------------------------------------------



```python
# 데이터 형태 확인
# train/test 둘다 첫번째 컬럼은 'id'이므로 제거
sample_sub=pd.read_csv('sample_submission.csv')
test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
# X,y 분리
X_train=train.iloc[:,1:41]
X_test=test.iloc[:,1:]
y_train=train.iloc[:,41:]
```


```python
X_train
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X00</th>
      <th>X01</th>
      <th>X02</th>
      <th>X03</th>
      <th>X04</th>
      <th>X05</th>
      <th>X06</th>
      <th>X07</th>
      <th>X08</th>
      <th>X09</th>
      <th>...</th>
      <th>X30</th>
      <th>X31</th>
      <th>X32</th>
      <th>X33</th>
      <th>X34</th>
      <th>X35</th>
      <th>X36</th>
      <th>X37</th>
      <th>X38</th>
      <th>X39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.7</td>
      <td>988.8</td>
      <td>1.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>1009.3</td>
      <td>989.6</td>
      <td>12.2</td>
      <td>1009.9</td>
      <td>1009.8</td>
      <td>...</td>
      <td>69.1</td>
      <td>8.2</td>
      <td>10.7</td>
      <td>1010.1</td>
      <td>0.00</td>
      <td>256.4</td>
      <td>0.0</td>
      <td>77.2</td>
      <td>62.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.3</td>
      <td>988.9</td>
      <td>1.7</td>
      <td>1.9</td>
      <td>0.0</td>
      <td>1009.3</td>
      <td>989.6</td>
      <td>12.1</td>
      <td>1010.0</td>
      <td>1009.9</td>
      <td>...</td>
      <td>70.3</td>
      <td>8.3</td>
      <td>10.3</td>
      <td>1010.1</td>
      <td>0.00</td>
      <td>215.4</td>
      <td>0.0</td>
      <td>77.3</td>
      <td>63.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.4</td>
      <td>989.0</td>
      <td>1.1</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>1009.2</td>
      <td>989.7</td>
      <td>12.1</td>
      <td>1010.1</td>
      <td>1010.1</td>
      <td>...</td>
      <td>71.5</td>
      <td>8.0</td>
      <td>9.7</td>
      <td>1010.0</td>
      <td>0.00</td>
      <td>235.2</td>
      <td>0.0</td>
      <td>77.3</td>
      <td>63.9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.4</td>
      <td>988.9</td>
      <td>1.5</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>1009.2</td>
      <td>989.6</td>
      <td>12.0</td>
      <td>1010.0</td>
      <td>1010.0</td>
      <td>...</td>
      <td>73.2</td>
      <td>7.7</td>
      <td>9.4</td>
      <td>1010.1</td>
      <td>0.00</td>
      <td>214.0</td>
      <td>0.0</td>
      <td>77.5</td>
      <td>64.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.2</td>
      <td>988.9</td>
      <td>0.8</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>1009.2</td>
      <td>989.7</td>
      <td>12.0</td>
      <td>1010.1</td>
      <td>1010.0</td>
      <td>...</td>
      <td>74.3</td>
      <td>7.4</td>
      <td>9.4</td>
      <td>1010.1</td>
      <td>0.00</td>
      <td>174.9</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>65.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4747</th>
      <td>19.9</td>
      <td>987.6</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>1006.9</td>
      <td>987.7</td>
      <td>21.7</td>
      <td>1007.5</td>
      <td>1007.4</td>
      <td>...</td>
      <td>89.9</td>
      <td>17.7</td>
      <td>19.1</td>
      <td>1007.5</td>
      <td>22.16</td>
      <td>218.6</td>
      <td>0.0</td>
      <td>82.3</td>
      <td>58.6</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4748</th>
      <td>19.9</td>
      <td>987.6</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>1006.8</td>
      <td>987.7</td>
      <td>21.6</td>
      <td>1007.5</td>
      <td>1007.4</td>
      <td>...</td>
      <td>91.3</td>
      <td>17.7</td>
      <td>19.2</td>
      <td>1007.5</td>
      <td>22.16</td>
      <td>161.7</td>
      <td>0.0</td>
      <td>82.5</td>
      <td>59.1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4749</th>
      <td>19.7</td>
      <td>987.7</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>1006.9</td>
      <td>987.6</td>
      <td>21.4</td>
      <td>1007.4</td>
      <td>1007.5</td>
      <td>...</td>
      <td>90.2</td>
      <td>17.8</td>
      <td>19.2</td>
      <td>1007.5</td>
      <td>22.16</td>
      <td>254.2</td>
      <td>0.0</td>
      <td>83.0</td>
      <td>58.9</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4750</th>
      <td>19.4</td>
      <td>987.7</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>1006.9</td>
      <td>987.8</td>
      <td>21.3</td>
      <td>1007.6</td>
      <td>1007.5</td>
      <td>...</td>
      <td>90.1</td>
      <td>17.7</td>
      <td>19.3</td>
      <td>1007.6</td>
      <td>22.16</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>83.2</td>
      <td>59.8</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>19.1</td>
      <td>987.6</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>1006.8</td>
      <td>987.8</td>
      <td>21.2</td>
      <td>1007.5</td>
      <td>1007.4</td>
      <td>...</td>
      <td>89.6</td>
      <td>17.7</td>
      <td>19.5</td>
      <td>1007.7</td>
      <td>22.16</td>
      <td>157.5</td>
      <td>0.0</td>
      <td>84.0</td>
      <td>59.5</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<p>4752 rows × 40 columns</p>
</div>




```python
y_train
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Y00</th>
      <th>Y01</th>
      <th>Y02</th>
      <th>Y03</th>
      <th>Y04</th>
      <th>Y05</th>
      <th>Y06</th>
      <th>Y07</th>
      <th>Y08</th>
      <th>Y09</th>
      <th>Y10</th>
      <th>Y11</th>
      <th>Y12</th>
      <th>Y13</th>
      <th>Y14</th>
      <th>Y15</th>
      <th>Y16</th>
      <th>Y17</th>
      <th>Y18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.5</td>
      <td>11.5</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>10.5</td>
      <td>10.0</td>
      <td>10.5</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>7.5</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.5</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.5</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>10.0</td>
      <td>9.5</td>
      <td>10.0</td>
      <td>9.5</td>
      <td>6.5</td>
      <td>7.5</td>
      <td>7.0</td>
      <td>8.5</td>
      <td>10.0</td>
      <td>9.5</td>
      <td>9.0</td>
      <td>7.5</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>19.5</td>
      <td>19.5</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>6.5</td>
      <td>7.5</td>
      <td>6.5</td>
      <td>8.0</td>
      <td>9.5</td>
      <td>9.5</td>
      <td>8.5</td>
      <td>7.5</td>
      <td>8.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>19.5</td>
      <td>19.5</td>
      <td>9.5</td>
      <td>9.0</td>
      <td>9.5</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>9.5</td>
      <td>9.0</td>
      <td>8.5</td>
      <td>7.5</td>
      <td>8.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.0</td>
      <td>10.5</td>
      <td>10.5</td>
      <td>19.0</td>
      <td>19.5</td>
      <td>9.5</td>
      <td>8.5</td>
      <td>9.5</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>7.5</td>
      <td>9.5</td>
      <td>9.0</td>
      <td>8.5</td>
      <td>7.5</td>
      <td>8.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4747</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>4748</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>4749</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>4750</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.0</td>
    </tr>
  </tbody>
</table>
<p>4752 rows × 19 columns</p>
</div>


학습시킬 Y data를 각각 분리
- Y00~Y17 30일치 
- Y18 3일치


```python
# Y00~Y17 30일치 
y_17=y_train.iloc[:4320,:18]
# Y18 3일치
y_18=y_train.iloc[4320:,18:19]
```


# 앙상블 모델
## 1. Model 1 : X00~39 -> Y00~17 학습하는 모델

```python
# ensemble 할 model 정의
models=[
    ('dt',DecisionTreeRegressor()),
    ('rf',RandomForestRegressor()),
    ('ab',AdaBoostRegressor()),
    ('br',BaggingRegressor()),
    ('gb',GradientBoostingRegressor()),
    ('sv',SVR()),
    ('lgbm',LGBMRegressor()),
    ('kn', KNeighborsRegressor()),
]
```

Base-Line 들 앙상블 후에 Y00~Y17 이 MultiOutput 이므로 MultiOutputRegressor 를 사용함.

```python
vote  = VotingRegressor(models) # Base-Line 들 앙상블
mr=MultiOutputRegressor(vote) 
mr.fit(X_train,y_17)
```


```python
#3일치 y17 예측
pred_y17_2=mr.predict(X_train.iloc[4320:,:])
pred_17_2_df=pd.DataFrame(pred_y17_2,index=list(range(4320,4752)),columns=y_train.columns[:-2])
pred_17_2_df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Y00</th>
      <th>Y01</th>
      <th>Y02</th>
      <th>Y03</th>
      <th>Y04</th>
      <th>Y05</th>
      <th>Y06</th>
      <th>Y07</th>
      <th>Y08</th>
      <th>Y09</th>
      <th>Y10</th>
      <th>Y11</th>
      <th>Y12</th>
      <th>Y13</th>
      <th>Y14</th>
      <th>Y15</th>
      <th>Y16</th>
      <th>Y17</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4320</th>
      <td>21.208813</td>
      <td>21.568323</td>
      <td>21.795442</td>
      <td>24.819350</td>
      <td>24.876024</td>
      <td>21.092153</td>
      <td>19.532982</td>
      <td>19.794123</td>
      <td>20.661040</td>
      <td>19.337865</td>
      <td>19.947454</td>
      <td>19.428595</td>
      <td>19.429348</td>
      <td>19.513226</td>
      <td>19.352403</td>
      <td>19.722278</td>
      <td>19.626023</td>
      <td>19.423035</td>
    </tr>
    <tr>
      <th>4321</th>
      <td>20.980285</td>
      <td>21.249925</td>
      <td>21.506281</td>
      <td>24.725453</td>
      <td>24.676404</td>
      <td>20.941252</td>
      <td>19.655279</td>
      <td>19.587861</td>
      <td>20.448768</td>
      <td>19.215744</td>
      <td>19.586013</td>
      <td>19.286929</td>
      <td>19.293463</td>
      <td>19.477894</td>
      <td>19.124550</td>
      <td>19.293079</td>
      <td>19.416224</td>
      <td>19.341358</td>
    </tr>
    <tr>
      <th>4322</th>
      <td>20.705078</td>
      <td>20.954702</td>
      <td>21.023796</td>
      <td>24.502623</td>
      <td>24.446196</td>
      <td>20.991917</td>
      <td>19.798831</td>
      <td>20.103680</td>
      <td>20.211813</td>
      <td>19.214107</td>
      <td>19.572131</td>
      <td>19.213822</td>
      <td>19.602044</td>
      <td>19.511678</td>
      <td>19.101478</td>
      <td>19.395488</td>
      <td>19.424228</td>
      <td>19.614542</td>
    </tr>
    <tr>
      <th>4323</th>
      <td>20.063013</td>
      <td>19.800407</td>
      <td>19.809692</td>
      <td>24.111867</td>
      <td>24.088272</td>
      <td>19.958550</td>
      <td>18.666257</td>
      <td>18.862618</td>
      <td>19.269271</td>
      <td>17.883972</td>
      <td>18.073073</td>
      <td>17.478343</td>
      <td>18.439007</td>
      <td>18.756071</td>
      <td>18.370867</td>
      <td>18.496772</td>
      <td>18.397410</td>
      <td>18.387222</td>
    </tr>
    <tr>
      <th>4324</th>
      <td>20.041240</td>
      <td>19.802235</td>
      <td>19.824365</td>
      <td>24.079082</td>
      <td>24.060690</td>
      <td>19.897376</td>
      <td>18.706306</td>
      <td>18.855275</td>
      <td>19.318533</td>
      <td>18.120383</td>
      <td>18.079732</td>
      <td>17.459197</td>
      <td>18.428259</td>
      <td>18.768299</td>
      <td>18.422688</td>
      <td>18.975808</td>
      <td>18.388632</td>
      <td>18.456520</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4747</th>
      <td>22.231192</td>
      <td>22.749325</td>
      <td>22.944365</td>
      <td>25.665338</td>
      <td>26.015563</td>
      <td>21.140964</td>
      <td>20.106047</td>
      <td>20.844484</td>
      <td>20.771054</td>
      <td>19.370331</td>
      <td>20.245422</td>
      <td>20.418929</td>
      <td>20.598240</td>
      <td>19.819910</td>
      <td>20.494114</td>
      <td>19.505393</td>
      <td>20.043739</td>
      <td>19.552402</td>
    </tr>
    <tr>
      <th>4748</th>
      <td>21.418708</td>
      <td>21.895753</td>
      <td>22.151912</td>
      <td>25.383738</td>
      <td>25.724320</td>
      <td>20.326478</td>
      <td>19.178999</td>
      <td>19.979171</td>
      <td>19.987046</td>
      <td>18.546963</td>
      <td>19.355602</td>
      <td>19.544983</td>
      <td>19.576850</td>
      <td>19.118983</td>
      <td>19.869316</td>
      <td>18.838983</td>
      <td>19.085929</td>
      <td>18.726659</td>
    </tr>
    <tr>
      <th>4749</th>
      <td>21.789445</td>
      <td>22.580554</td>
      <td>22.557074</td>
      <td>25.688781</td>
      <td>26.072885</td>
      <td>21.077769</td>
      <td>20.019075</td>
      <td>20.837508</td>
      <td>20.459646</td>
      <td>19.418346</td>
      <td>20.186591</td>
      <td>20.328738</td>
      <td>20.473843</td>
      <td>19.526403</td>
      <td>20.418839</td>
      <td>19.524575</td>
      <td>19.867764</td>
      <td>19.525386</td>
    </tr>
    <tr>
      <th>4750</th>
      <td>21.014497</td>
      <td>21.331811</td>
      <td>21.522001</td>
      <td>25.812836</td>
      <td>25.754004</td>
      <td>19.874959</td>
      <td>18.648872</td>
      <td>19.370050</td>
      <td>19.280055</td>
      <td>18.080009</td>
      <td>18.650392</td>
      <td>18.669360</td>
      <td>18.924435</td>
      <td>18.851851</td>
      <td>18.705127</td>
      <td>18.462711</td>
      <td>18.484844</td>
      <td>18.112578</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>20.608769</td>
      <td>20.816668</td>
      <td>20.760233</td>
      <td>25.289103</td>
      <td>25.224812</td>
      <td>19.328851</td>
      <td>18.327453</td>
      <td>19.083198</td>
      <td>18.779052</td>
      <td>17.683622</td>
      <td>18.244295</td>
      <td>18.144811</td>
      <td>18.575555</td>
      <td>18.347631</td>
      <td>18.194029</td>
      <td>18.230219</td>
      <td>18.091944</td>
      <td>17.843798</td>
    </tr>
  </tbody>
</table>
<p>432 rows × 18 columns</p>


## 2. Model 2 : Y00~17 -> Y18  학습하는 모델

```python
# 3일치 Y들로 Y18 학습
vote2  = VotingRegressor(models)
mr2=MultiOutputRegressor(vote2)
mr2.fit(pred_17_2_df,y_18)
```



# 테스트 데이터로 예측하기
## Model 1 : X_test -> Y00~17 예측

```python
pred_y17_test=mr.predict(X_test)
pred_y17_test_df=pd.DataFrame(pred_y17_test)
```

## 2. Model 2 : Y00~17 -> Y18  얘측
```python
pred_y18_test=mr2.predict(pred_y17_test_df)
pred_y18_test_df=pd.DataFrame(pred_y18_test)
pred_y18_test_df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.424656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.596596</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.615842</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.347743</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.653030</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>11515</th>
      <td>26.542946</td>
    </tr>
    <tr>
      <th>11516</th>
      <td>26.870129</td>
    </tr>
    <tr>
      <th>11517</th>
      <td>26.762396</td>
    </tr>
    <tr>
      <th>11518</th>
      <td>26.573369</td>
    </tr>
    <tr>
      <th>11519</th>
      <td>26.562976</td>
    </tr>
  </tbody>
</table>
<p>11520 rows × 1 columns</p>




```python
pred_y18_test_df.to_csv('test_pred.csv')
```


-----------------------------
## 모델 수정
4월 26일에는 아래 모델 추가.

```python
('ridge',Ridge()),
('lasso',Lasso())
```
수정한 모델로 예측한 예측값

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.960984</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.270790</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.988945</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.886526</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.643114</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>11515</th>
      <td>26.538323</td>
    </tr>
    <tr>
      <th>11516</th>
      <td>26.889662</td>
    </tr>
    <tr>
      <th>11517</th>
      <td>26.821821</td>
    </tr>
    <tr>
      <th>11518</th>
      <td>26.285530</td>
    </tr>
    <tr>
      <th>11519</th>
      <td>26.424193</td>
    </tr>
  </tbody>
</table>
<p>11520 rows × 1 columns</p>


--------------------------------------------------------------------------------------------------

## 결과 (가채점 점수 기준)
팀 합류 전 : 2020-03-06 8.12<br> 
팀 합류 후 
- (앙상블 모델)<br>2020-03-20  :  4.87<br>
- (Ridge,Lasso 추가 모델)<br>2020-04-06 : 4.58 (Ridge,Lasso 추가 모델)

## 아쉬운 점
- 시계열 특성을 잘 고려하지 못함.
Y값들이 경향이 보였는데 X와 Y 데이터의 전처리 방향을 못잡음.

- Train 데이터의 사이즈가 5천개도 되지않아 과적합 우려.
특히 Model 2 를 학습시킬 때 Y값들이 3일치 데이터만으로 Y18을 학습했기 때문에 이 부분에 대한 보완점이 필요.  
