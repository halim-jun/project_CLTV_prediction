# project_CLTV_prediction

## 1. 해커톤 3차 제출용 파일 : 

- 대부분의 EDA, Machine learning model 구축 과정을 설명해두었습니다.

## 2. python for machine learning operation with AWS s3
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

