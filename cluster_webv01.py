import streamlit as st
# import module spotifyDataExtract
# import spotifyDataExtract as sde
# import model from pickle
import pickle
# untuk mengatur data
import pandas as pd
import numpy as np
# data visualiation
import matplotlib.pyplot as plt
import seaborn as sns
# machine learning library with its metrics
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# from pydub import AudioSegment

##################### NARIK DATA #####################
# lagu Kpop dari Playlist "K-Pop Hits 2021" pada user Filtr Éxitosp
# kpop1_song = sde.get_playlist_tracks(username='spotify:user:mejoresplaylistsspotify', 
#                                     playlist_id='spotify:playlist:4QXgG8a9KkUzaNpHS98G0O')
# lagu Kpop dari Playlist "TOP 100 KPOP Songs 2021" pada user David Li
# kpop2_song = sde.get_playlist_tracks(username='spotify:user:1158834068', 
#                                     playlist_id='spotify:playlist:5Di5zq6FdHlrRdcqJxfVmA')     

# tarik data dari masing-masing playlist track
# df_kpop1 = sde.get_data_from_tracks(kpop1_song)
# df_kpop2 = sde.get_data_from_tracks(kpop2_song)

# combine and sort dataset
# lagu_kpop = sde.append_sort_df(df_kpop1, df_kpop2)
lagu_kpop = pd.read_csv('lagu_kpop.csv')

##################### DATA PREPARATION & MODEL INFERENCE ##################### 
# fixed preparation
mm = MinMaxScaler()
mm.fit(lagu_kpop[['tempo', 'loudness']])
lagu_kpop[['tempo', 'loudness']] = mm.transform(lagu_kpop[['tempo', 'loudness']])

# fitur yang dipakai
song_features = lagu_kpop[['danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness']].values

# load kmeans model
kmeans_model = pickle.load(open('kmeans_model.pickle', 'rb'))
model_label = kmeans_model.predict(song_features)

# masukan kmeans label ke data
lagu_kpop['cluster'] = model_label
# view The closest song from Each Cluster Centroid
closest, _ = metrics.pairwise_distances_argmin_min(kmeans_model.cluster_centers_, song_features)

##################### DASHBOARD DISPLOT VISUALIZATION #####################

def distributionplot(data, features, title):
    fig, ax1 = plt.subplots(figsize=(20,10))
    sns.distplot(data[features], label = 'K-Pop')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(title, fontsize = 24)
    plt.legend()
    st.pyplot(fig)
    st.write(f'''Mean of {features} is {data[features].mean():.4f} and the Standard Deviation of {features} is {data[features].std():.4f}''')

# visualisasi distribusi cluster
def distributioncluster(data, features, title):
    fig, ax1 = plt.subplots(figsize=(20,10))
    sns.distplot(data[data['cluster'] == 0][features], label = 'Cluster 1')
    sns.distplot(data[data['cluster'] == 1][features], label = 'Cluster 2')
    sns.distplot(data[data['cluster'] == 2][features], label = 'Cluster 3')
    sns.distplot(data[data['cluster'] == 3][features], label = 'Cluster 4')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(title, fontsize = 24)
    plt.legend()
    st.pyplot(fig)


######## STREAMLIT APP ######## 
st.write("""
# Korean Pop Song Clustering
song with closest centroid from each cluster
""")
# tampilkan lagu yang centroidnya terdekat
lagu_kpop.loc[closest,['name', 'artist', 'cluster']]
st.write("""
## Song Features Distributions
""")

distributionplot(lagu_kpop, 'acousticness', 'Acousticness')
distributionplot(lagu_kpop, 'danceability', 'Danceability')
distributionplot(lagu_kpop, 'energy', 'Energy')
distributionplot(lagu_kpop, 'valence', 'Valence')
distributionplot(lagu_kpop, 'tempo', 'Tempo')
distributionplot(lagu_kpop, 'loudness', 'Loudness')

st.write("""## Song Features Distributions Distributions based on each Cluster""")
distributioncluster(lagu_kpop, 'acousticness', 'Acousticness')
distributioncluster(lagu_kpop, 'danceability', 'Danceability')
distributioncluster(lagu_kpop, 'energy', 'Energy')
distributioncluster(lagu_kpop, 'valence', 'Valence')
distributioncluster(lagu_kpop, 'tempo', 'Tempo')
distributioncluster(lagu_kpop, 'loudness', 'Loudness')

##################### UPLOAD FILE ##################### 

# def csv_to_singlerow(song_file, normalized=False):
#     """MP3 to a row of data"""
#     sound = AudioSegment.from_mp3(song_file)
#     sound.export("song_file.wav",format="wav")
#     y = np.array(sound.get_array_of_samples())
#     if sound.channels == 2:
#         y = y.reshape((-1, 2))
#     if normalized:
#         return sound.frame_rate, np.float32(y) / 2**15
#     else:
#         return sound.frame_rate, y

def predict(song_data, model):   
    model_label = model.predict(song_data)    
    # masukan kmeans label ke data
    song_data['cluster'] = model_label
    predict_label = song_data['cluster']
    return predict_label

# song_upload = st.file_uploader("Please Upload Mp3 Audio File Here !", type=["mp3"])
song_upload = st.file_uploader("Please Upload Mp3 Audio File Here !", type=["csv"])
if song_upload is None:
    st.write("Please upload MP3 File !!")
else:
    ######## DISPLAY OUTPUT OF UPLOADED SONG ########  
    song_uploaded = pd.read_csv(song_upload)
    song_data = song_uploaded[['danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness']]
    song_cluster = predict(song_data, kmeans_model)   
    st.write("## The Cluster of Song is:")
    st.write(song_cluster)
