# project_CLTV_prediction

## 1. 프로젝트 개요 (Introduction) : 

- 고객의 기존 거래 내역을 통해 다음 월의 구매 수치를 예측합니다. (The project aims to build a model that predict next month customers' transaction volume using the last two month data)
- Commerce 사의 상거래 데이터를 이용 (Utilized Korean Commerce company's transaction data)
- 해당 모델을 고객의 구매 금액을 선 파악하여, 마케팅 비용을 최적화 하고 구매 액수가 감소할 고객을 미리 파악하여 선제적 대응을 취하는 비즈니스 전략 수행을 목표로 하고 있습니다. (The model aims to optimize marketing cost and take preemptive marketing strategy by figuring out customers who's transaction volume is expected to decrease or increase)
- 데이터 : 국내 e commerce 사 데이터 (Data : Korean Commerce company data)

## 2. Final_product.ipynb : 

- 대부분의 EDA, Machine learning model 구축 과정을 자세히 설명해두었습니다. (Most EDA and process for developing model are explained in this file)
- 해당 노트에는 세부적인 코드가 포함되어 있으므로 간략한 설명을 위해서는 "Project Description (Eng).pdf"/ "프로젝트 설명 (Kor).pdf" 을 참고해주세요!
(This note contains detailed code : for better explanation refer to the "Project Description (Eng).pdf"

## 3. python for machine learning operation with AWS s3
- retrieve_data.py
: AWS 에 데이터를 올리거나 내려받는 boto3 패키지를 활용한 코드 입니다. 해당 access key 는 현재 보안상의 이유로 deactivate 해 두었습니다.(Code that interacts with Amazon Web Service cloud database (S3). The access key has been deactivated for security reasons)
- preprocess.py
: 머신러닝모델을 돌리기 위해서 데이터를 전처리하는 파일입니다. (Code that preprocess raw data)
- model.py
: 모델을 돌리고 해당 모델의 weight나 결과값을 return 하는 파일입니다. (Code that runs the model and returns model weight and result)
- prediction_result.py
: 저장되어있는 weight 를 이용하여 모델의 성능을 구체적으로 확인하고 결과값을 저장하는 파일입니다. (Code that specifies model performance and returns the outcome)
- execution.py
: 위의 일련의 과정을 수행하여 클라우드 상의 모델 다운로드 -> 머신러닝 수행 -> 결과값 도출 ->weight 클라우드에 재 업로드 시킵니다. (This code runs through all the machine learning operation process stated above. It downloads data from the s3 cloud -> Model learning -> Returns the outcome -> Reupload the model weight to the AWS cloud)

# 4. Project Description

- For easy to understand explanation of the project, please kindly refer to the "Project Description (Eng).pdf" or "프로젝트 설명 (Kor).pdf"

