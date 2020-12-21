import retrieve_data
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import datetime
import os
def common_preprocess():
    trans=retrieve_data.download_data()
    trans=trans[trans['buy_am']>0]
    trans.loc[trans['buy_ct']==0, 'buy_ct']=1
    def rename(data, origin, to):
        data.rename(columns={origin:to}, inplace=True)
        return data
    def merge(a, b, how):
        x=pd.merge(a, b, how=how, left_on='clnt_id', right_on='clnt_id')
        return x
    rename(trans, 'buy_am', 'total_price')
    trans["itm_price"]=trans['total_price']/trans['buy_ct']
    #이상치 제거 전처리
    too_high_payer=trans.groupby('trans_id').sum()['total_price'].sort_values(ascending=False).head(13).index

    transcleaned=trans[~trans['trans_id'].isin(too_high_payer)]
    too_low_payer=trans.groupby('trans_id').sum()['total_price'].sort_values().head(1000).index
    transcleaned=transcleaned[~trans['trans_id'].isin(too_low_payer)]
    trans=transcleaned
    #데이터 중복제거
    trans.drop_duplicates(inplace=True)
    #날짜 데이터 전처리 
    trans['de_dt']=trans['de_dt'].astype('str')
    trans['de_tm']=trans['de_tm'].astype('str')
    trans['de_tm']=pd.to_datetime(trans['de_dt'].str[0:4]+trans['de_dt'].str[4:7]+trans['de_dt'].str[7:]+"-"+trans['de_tm'])
    trans['month']=trans['de_tm'].dt.month
    trans['date']=trans['de_tm'].dt.date
    
    #개별 월별 구매내역 
    jul=trans[trans['de_tm'].dt.month==7]
    aug=trans[trans['de_tm'].dt.month==8]
    sep=trans[trans['de_tm'].dt.month==9]
    sum_jul=pd.pivot_table(data=jul,index='clnt_id', values=['total_price', 'buy_ct'], aggfunc='sum', columns='biz_unit')
    mean_jul=pd.pivot_table(data=jul,index='clnt_id', values=['total_price', 'buy_ct', ], aggfunc='mean', columns='biz_unit')
    # mean_jul=pd.pivot_table(data=jul,index='clnt_id', values=['total_price', 'buy_ct','item_price' ], aggfunc='mean', columns='biz_unit').reset_index()
    mean_jul.fillna(0, inplace=True)
    count_jul=pd.pivot_table(data=jul,index='clnt_id', values='trans_id', aggfunc=pd.Series.nunique, columns='biz_unit')

    rename(sum_jul, 'buy_ct', 'jul_itm_sum')
    rename(sum_jul, 'total_price', 'jul_sum_price')
    rename(mean_jul, 'buy_ct', 'jul_itm_mean')
    rename(mean_jul, 'total_price', 'jul_mean_price')
    rename(count_jul, 'trans_id', 'jul_trans_cnt')
    jul_x=merge(merge(sum_jul, mean_jul, 'inner'), count_jul, 'inner')
    sum_aug=pd.pivot_table(data=aug,index='clnt_id', values=['total_price', 'buy_ct'], aggfunc='sum', columns='biz_unit')
    sum_aug.fillna(0, inplace=True)
    mean_aug=pd.pivot_table(data=aug,index='clnt_id', values=['total_price', 'buy_ct', ], aggfunc='mean', columns='biz_unit')
    # mean_aug=pd.pivot_table(data=aug,index='clnt_id', values=['total_price', 'buy_ct','item_price' ], aggfunc='mean', columns='biz_unit').reset_index()
    mean_aug.fillna(0, inplace=True)
    count_aug=pd.pivot_table(data=aug,index='clnt_id', values='trans_id', aggfunc=pd.Series.nunique, columns='biz_unit')
    rename(sum_aug, 'buy_ct', 'aug_itm_sum')
    rename(sum_aug, 'total_price', 'aug_sum_price')
    rename(mean_aug, 'buy_ct', 'aug_itm_mean')
    rename(mean_aug, 'total_price', 'aug_mean_price')
    rename(count_aug, 'trans_id', 'aug_trans_cnt')
    aug_x=merge(merge(sum_aug, mean_aug, 'inner'), count_aug, 'inner')
    data=merge(merge(aug_x, jul_x, 'inner'), sep.groupby('clnt_id').sum()['total_price'], "inner")
    rename(data, 'total_price', 'sep_sum_price')
    data.fillna(0, inplace=True)
    data=data.reset_index()
    #각 거래마다의 deviation
    jul_aug=trans[trans['de_tm'].dt.month.isin([7,8])]
    df=jul_aug.groupby(["clnt_id","trans_id"]).sum()['total_price']
    sf=pd.DataFrame(df)
    ssf=sf.groupby('clnt_id').std()
    data=merge(data, ssf,'inner')
    rename(data, 'total_price', 'each_transaction_deviation')
    #7,8월 구매액 합계
    data=merge(data, trans[trans['de_tm'].dt.month.isin([7,8])].groupby('clnt_id').sum()['total_price'], 'inner')
    rename(data, 'total_price', 'jul_aug_price')
    #7월과 8월의 거래 금액 차이
    df=np.square(sum_aug['aug_sum_price']-sum_jul['jul_sum_price'])/(sum_aug['aug_sum_price']+sum_jul['jul_sum_price'])
    df.fillna(0,inplace=True)
    price_randomness=df['A01']+df['A02']+df['A03']+df['B01']+df['B02']+df['B03']
    price_randomness=pd.DataFrame(price_randomness)
    price_randomness.rename(columns={0:'price_randomness_square'}, inplace=True)
    price_randomness=np.log(price_randomness+1)
    data=merge(data, price_randomness, 'inner')
    data['low_randomness']=0
    data.loc[data['price_randomness_square']<13, 'low_randomness']=1
    data['high_randomness']=0
    data.loc[data['price_randomness_square']>=13, 'high_randomness']=1
    data['ft_cross_aug_jul+low_rand']=data['low_randomness']*data['jul_aug_price']
    data['ft_cross_aug_jul+high_rand']=data['high_randomness']*data['jul_aug_price']
  
    #recency

    df1=trans[trans['month'].isin([7,8])]
    df=df1.groupby('clnt_id').max()['date']
    df=pd.DataFrame(df)
    df['now']=pd.to_datetime("2019-08-30")
    df['date']=pd.to_datetime(df['date'])
    df['recency']=df['now']-df['date']
    df['recency']=df['recency'].astype(int)
    df
    #frequencey
    freq=pd.DataFrame((trans.groupby('clnt_id').max()['de_tm']-trans.groupby('clnt_id').min()['de_tm'])/trans.groupby('clnt_id').nunique()['trans_id'])
    freq.columns=['freq']
    freq=freq.astype(int)
    freq
    df1=pd.merge(df, freq, left_index=True, right_index=True)
    df1['recency']=df1['recency']+1
    df1['activeness']=df1['freq']/df1['recency']+1

    #merge
    df1[['recency', 'freq', 'activeness']]
    data=merge(data, df1[['recency', 'freq', 'activeness']], "inner")
    data['active_group']=0
    data['non_active_group']=0
    data.loc[data['activeness']<8, 'non_active_group']=1
    data.loc[data['activeness']>=8, 'active_group']=1
    #activeness, randomness, jul_aug_price cross feature
    data['cross_nonac_julaug_lowrand']=np.log(data['non_active_group']*data['jul_aug_price']*data['low_randomness']+1)
    data['cross_nonac_julaug_highrand']=np.log(data['non_active_group']*data['jul_aug_price']*data['high_randomness']+1)
    data['cross_ac_julaug_lowrand']=np.log(data['active_group']*data['jul_aug_price']*data['low_randomness']+1)
    data['cross_ac_julaug_highrand']=np.log(data['active_group']*data['jul_aug_price']*data['high_randomness']+1)
    #장바구니 내 제품 다양성  product_diversity_ratio
    trans.loc[trans['pd_c']=="unknown", 'pd_c']=0
    df=trans[trans['month'].isin([7,8])].groupby(['clnt_id','pd_c']).sum()['buy_ct']/trans[trans['month'].isin([7,8])].groupby('clnt_id').sum()['buy_ct']
    df1=np.square(df)
    df2=df1.groupby('clnt_id').sum()
    df2.columns=['product_diversity_ratio']
    data=merge(data, np.log(df2), "inner")
    #제품 다양성
    train_trans=trans[trans['month'].isin([7,8])]
    data=merge(data,np.log(train_trans.groupby('clnt_id').nunique()['pd_c']+1), 'inner')
    rename(data, 'pd_c', 'product_div_count')
    data.rename(columns={'buy_ct_y':'product_diversity_ratio'}, inplace=True)
    return data


def preprocess_for_model():
    
    data=common_preprocess()
    data_X=data.drop(['clnt_id', 'sep_sum_price'],axis=1)
    data_Y=np.log(data['sep_sum_price']+1)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_X=pd.DataFrame(scaler.fit_transform(data_X))
    X_train, X_test, Y_train, Y_test=train_test_split(data_X, data_Y, test_size=0.2,random_state=0)

    partial_x_train, x_val, partial_y_train, y_val=train_test_split(X_train, Y_train, test_size=0.2,random_state=0)
    return partial_x_train,partial_y_train,x_val,y_val,X_test,Y_test

# def preprocess_for_eda():

    # data preprocess
    # return 
#순서대로 실행할것
def save_preprocessed_data():
    data=common_preprocess()
    now=datetime.datetime.now()
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    path=os.getenv('HOME')+'/aiffel/kaggle/Lpoint/preprocessed_data/'
    data.loc[data['jul_aug_price']<100000*2, 'crnt_level']=0
    data.loc[data['jul_aug_price']>=100000*2, 'crnt_level']=1
    data.loc[data['jul_aug_price']>=250000*2, 'crnt_level']=2
    data.loc[data['jul_aug_price']>=500000*2, 'crnt_level']=3
    data.loc[data['jul_aug_price']>=800000*2, 'crnt_level']=4
    data.loc[data['jul_aug_price']>=1500000*2, 'crnt_level']=5
    data.to_csv(path+date_time+'preprocessed_data.csv')
    return data
