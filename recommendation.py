import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


data_sample=[]
def restaurant_recommendation(location, title):
    global data_sample       
    global cosine_sim
    global sim_scores
    global tfidf_matrix
    global corpus_index
    global feature
    global rest_indices
    global idx
    
    
    data=pd.read_csv('zomato.csv', encoding='latin1')
    is_indian_restaurants=data['Country Code']==1
    indian = data[is_indian_restaurants]
    is_delhi = data['City']=='New Delhi'
    delhi_restaurants = indian[is_delhi]
    data_minimal=delhi_restaurants[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Locality', 'Aggregate rating']]
    data_minimal['Locality'].value_counts(dropna=False)
    
    
    data_sample=data_minimal.loc[data_minimal['Locality']==location]
    data_sample.reset_index(level=0, inplace=True)
    data_sample['Split']="X"
    for i in range(0, data_sample.index[-1]):
        split_data=re.split(r'[,]', data_sample['Cuisines'][i])
        for k,l in enumerate(split_data):
            split_data[k]=(split_data[k].replace(" ", ""))
        split_data=' '.join(split_data[:])
        data_sample['Split'].iloc[i]=split_data
    tfidf=TfidfVectorizer(stop_words='english')
    data_sample['Split']=data_sample['Split'].fillna('')
    tfidf_matrix=tfidf.fit_transform(data_sample['Split'])
    tfidf_matrix.shape
    feature=tfidf.get_feature_names()
    
    cosine_sim=linear_kernel(tfidf_matrix, tfidf_matrix)
    corpus_index=[n for n in data_sample['Split']]
    indices=pd.Series(data_sample.index, index=data_sample['Restaurant Name']).drop_duplicates()
    idx=indices[title]
    sim_scores=[]
    for i,j in enumerate(cosine_sim[idx]):
        k=data_sample['Aggregate rating'].iloc[i]
        if j != 0 :
            sim_scores.append((i,j,k))
    sim_scores = sorted(sim_scores, key=lambda x: (x[1],x[2]) , reverse=True)
    sim_scores = sim_scores[0:10]
    
    rest_indices=[i[0] for i in sim_scores]
    data_x=data_sample[['Restaurant Name','Aggregate rating']].iloc[rest_indices]
    
    data_x['Cuisines']= data_sample['Cuisines']
    data_x['Average Cost for two']= data_sample['Average Cost for two']
   
    return data_x