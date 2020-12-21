# project_CLTV_prediction

## 1. 프로젝트 개요 : 

- 고객의 기존 거래 내역을 통해 다음 월의 구매 수치를 예측합니다.
- Commerce 사의 상거래 데이터를 이용 
- 해당 모델을 고객의 구매 금액을 선 파악하여, 마케팅 비용을 최적화 하고 구매 액수가 감소할 고객을 미리 파악하여 선제적 대응을 취하는 비즈니스 전략 수행을 목표로 하고 있습니다.
- 데이터 : 국내 e commerce 사 데이터

## 2. 해커톤 3차 제출용 파일 : 

- 대부분의 EDA, Machine learning model 구축 과정을 설명해두었습니다.

## 3. python for machine learning operation with AWS s3
- retrieve_data.py
: AWS 에 데이터를 올리거나 내려받는 boto3 패키지를 활용한 코드 입니다. 해당 access key 는 현재 보안상의 이유로 deactivate 해 두었습니다.
- preprocess.py
: 머신러닝모델을 돌리기 위해서 데이터를 전처리하는 파일입니다.
- model.py
: 모델을 돌리고 해당 모델의 weight나 결과값을 return 하는 파일입니다.
- prediction_result.py
: 저장되어있는 weight 를 이용하여 모델의 성능을 구체적으로 확인하고 결과값을 저장하는 파일입니다. 
- execution.py
: 위의 일련의 과정을 수행하여 클라우드 상의 모델 다운로드 -> 머신러닝 수행 -> 결과값 도출 ->weight 클라우드에 재 업로드 시킵니다.

