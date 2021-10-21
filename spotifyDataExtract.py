# package spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data

# untuk mengatur data
import pandas as pd

client_id = 'client id'
client_secret = 'client password'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                    client_secret=client_secret)
# spotify object to access API
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 

def get_playlist_tracks(username,playlist_id):
    """
    Function to extract data from playlist more than 100 rows 
    by accessing spotify object based on client credential manager
    
    Args: 
        username(string) : username of spotify account selected
        playlist_id(string) : represent as playlist selected in its spotify account

    Returns: 
        list : list track of songs collected from playlist of selected account
    """
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


# create function to transform playlist into song dataframe
def get_data_from_tracks(tracks):        
    """
    Function to extract data from tracks, 
    with Dict to contain data extracted temporary
    Then iterate each index from list of tracks
    
    Args: 
        tracks (list) : list track of songs collected from playlist of its spotify account
    

    Returns: 
        dataframe: dataset that contained extracted song from playlist
    """
    # buat dict kosong untuk tiap track playlist 
    dict_data = {}
    dict_data['name'] = []
    dict_data['artist'] = []
    dict_data['uri'] = []
    dict_data['key'] = []
    dict_data['mode'] = []
    dict_data['acousticness'] = []
    dict_data['danceability'] = []
    dict_data['energy'] = []
    dict_data['liveness'] = []
    dict_data['loudness'] = []
    dict_data['tempo'] = []
    dict_data['valence'] = []
    dict_data['popularity'] = []

    #  tarik data dari masing-masing playlist lagu 
    for i in tracks:
        print(i['track']['name'])
        try:
            features = sp.audio_features(i['track']['uri'])
        except:
            print(i['track']['uri'])
            print('skipped')
            continue
        # kemudian di isi dengan key value nya
        dict_data['name'].append(i['track']['name'])
        dict_data['artist'].append(i['track']['artists'][0]['name'])
        dict_data['uri'].append(i['track']['uri'])
        dict_data['key'].append(features[0]['key'])
        dict_data['mode'].append(features[0]['mode'])
        dict_data['acousticness'].append(features[0]['acousticness'])
        dict_data['danceability'].append(features[0]['danceability'])
        dict_data['energy'].append(features[0]['energy'])
        dict_data['liveness'].append(features[0]['liveness'])
        dict_data['loudness'].append(features[0]['loudness'])
        dict_data['tempo'].append(features[0]['tempo'])
        dict_data['valence'].append(features[0]['valence'])
        dict_data['popularity'].append(i['track']['popularity'])

    df_dict = pd.DataFrame(dict_data)
    return df_dict

# append dan urutkan dataframe agar lebih rapih
def append_sort_df(df1, df2):
    """
    Function to combine dataset then sort the dataset 
    
    Args: 
        df1(DataFrame) : represent as first dataset to be combined with second dataset
        df2(DataFrame) : represent as second dataset as ingredient
    
    Function:
        
        append(dataframe, ignore_index) 
            to combine dataset the dataset vertically with matching column each other
            parameter : 
                dataframe, dataset to be combined with first data
                ignore_index, to ignoring index (default False)

        sort_values(subset='column', ascending=False)
            to sort dataset values based selected column in subset
            parameter : 
                subset, selected column to be sorted values
                ascending, to sort data from small to big (default True)

    Returns: 
        dataframe: final form dataframe as finished product 
    """
    # combine the dataset with append()
    df = df1.append(df2, ignore_index = True)
    # sort the dataset with sort_values() after combined
    df = df.sort_values('popularity', ascending=False).drop_duplicates('name').sort_index()
    return df