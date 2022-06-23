# -*- coding: utf-8 -*-
"""
Implements the section 4.3 Jupyter notebook function

REMARK: Note that the song of interest should be in the last position of the Spotify playlist

REMARK: The song singal is considered to come from two channels.
        As explained in https://github.com/tyiannak/pyAudioAnalysis/issues/162
        one option is take the channels average. In the code this is implemented by
        the stereo_to_mono function.

"""
import sys
import numpy as np
import pickle
import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as mF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import spatial
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


###############################################################################

CLIENT_ID = '1def38f0e05e4641b3b4dcd0edb4d720'
CLIENT_SECR = 'c0ee7748ea8b48ebb9b982bfe02e60e7'


def prepare(wav_path, track_uri):
    
    spotify_features = {k: sp.audio_features(track_uri)[0][k] for k in ('danceability', 'energy', 'loudness', 'mode', 'speechiness', 
                                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo')}
    song_spotify_features = np.array(list(spotify_features.values())) # shape is (10,)
    
    # extract pyaudio features
    m_win, m_step, s_win, s_step = 5, 5, 1, 1 
    fs, signal = aIO.read_audio_file(os.path.join(wav_path))
    signal = aIO.stereo_to_mono(signal)
    mt, st, feature_names = mF.mid_feature_extraction(signal, fs, 
                                                  m_win * fs, m_step * fs,
                                                  s_win * fs, s_step * fs)
    song_pyaudio_features = 0
    for j in range(mt.shape[1]):
        song_pyaudio_features += mt[:,j]
    song_pyaudio_features = song_pyaudio_features / mt.shape[1] # # shape is (136,)
    
    # put together the song spotify and pyaudio features
    song_features = np.concatenate((song_spotify_features, song_pyaudio_features), axis=0) # shape is (146,)
    song_features = np.reshape(song_features, (1, song_features.shape[0])) # shape is (1, 146)
    
    # pickle-load spotify+pyuadio features of the database files
    with open("all_features.pickle", "rb") as f: all_features = pickle.load(f)    
    band_song = []
    for j in range(170):
        band_song.append(all_features[j][0] + ' - ' + all_features[j][1])
    database_features = []
    for j in range(170):
        database_features.append(all_features[j][2])
    database_features = np.array(database_features) # shape is (170,146)
    
    # put together the song and the database features
    features = np.concatenate((song_features, database_features), axis = 0) # shape is (171,146)
    
    return features

###############################################################################

def get_similar(song_database_features):

    no_of_songs = 5 #no of similar songs to return

    with open("all_features.pickle", "rb") as f: all_features = pickle.load(f)    
    band_song = []
    for j in range(170):
        band_song.append(all_features[j][0] + ' - ' + all_features[j][1])

    # get pyaudio features, transform them in the direction of the first 3 principal components, 
    # and compute euclidean similarities matrix
    pyaudio_features = []
    for j in range(len(all_features)):
        pyaudio_features.append(all_features[j][2][:]) # since the first 10 features are the spotify features
        # len(pyaudio_features) = 170
        # len(pyaudio_features[0]) = 136
    
    scaler = StandardScaler()
    pca = PCA(n_components=25) # choose first 25 principal components to get visualiation results as well
    ar = np.array(scaler.fit_transform(pyaudio_features))
    transformed = pca.fit_transform(ar) # shape is (170,20)
    #print('The total explained variance ratio is {}'.format(round(np.sum(pca.explained_variance_ratio_),2)))
    # Due to the evaluation function in 4.4 we do not print the total explained variance ratio. We note that it is 1.

    size = transformed.shape[0]
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i<j:
                matrix[i,j] = spatial.distance.cosine(transformed[i], transformed[j])
                matrix[j,i] = matrix[i,j]

    # below code computes the values after the smallest one (i.e. 0) and their indices in the 
    index = 0
    values = np.sort(matrix[index, :], axis = 0)[1:(no_of_songs+1)]
    indices = matrix[index, :].argsort()[1:(no_of_songs+1)]

    # save the songs and scores in lists
    similar_songs = []; similar_scores = [];
    for j in range(no_of_songs):
        similar_songs.append(band_song[indices[j]])
        similar_scores.append(round(values[j],2))

    return similar_songs, similar_scores
    
if __name__ == '__main__':
    song_info = []

    for file in os.listdir(os.getcwd()):
        if file.endswith(".wav"):
            mod_file = file.split('.')[0]
            song_info = mod_file.strip().split('-')

    print(f'Looking for recommendations for {song_info[1]} by {song_info[0]}...')
   
    #Connect to Spotify's API
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECR)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    
    #Search for the track and find its uri
    track_id = sp.search(q='artist:' + song_info[0] + ' track:' + song_info[1], type='track', limit = 1)
    uri = track_id['tracks']['items'][0]['uri']

    feature_vect = prepare(file, uri)
    recommendations, _ = get_similar(feature_vect)
    print('Top-5 recommendations:')
    for i in range(5):
        print(f'{i+1}: {recommendations[i]}')


