import model
import retrieve_data
weight_path=model.model_update()[2]
retrieve_data.upload_model_weight(weight_path, "projectltv")
#이 코드면 클라우드에서 S3에서 최신의 데이터를 가지고 와서 전처리와 머신러닝을 수행합니다. 
# 그리고 머신러닝의 weight 를 Json file 로 변환하여 클라우드에 재 업로드 합니다.