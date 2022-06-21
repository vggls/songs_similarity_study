# -*- coding: utf-8 -*-
"""
For any wav file of interest below functions implement the 
- manhattan metric similarity (as introduced in section 4.1 of the Jupyter notebook)
- pyaudio metric similarity (as introduced in section 4.3 of the Jupyter notebook)
with respect to the 170 database songs.

More specifically:
    
* The "get_similar_manhattan_metric" function gets as input a wav file path, the user's
  Spotify credentials (client_id, secret_id) and the Spotify playlist link that includes
  the song of interest in the last (!!!) position.
  At first it computes the songs spotify (10) and pyaudio(136) features and then 
  concatenates them in a 146-dim vector. This is used to compute the manhattan distances 
  of all database songs (their features are included in the all_features.pickle file).
  Eventually, the 5 closest database songs are selected.
  
* The "get_similar_pyaudio_metric" gets as input a wav file path 
  and at first computes its pyaudio(136) features. 
  In addition, it loads the database song features (includes in all_features.pickle) and 
  transforms them in the direction of their first two principal components.
  Then the selected song features are transformed as well and eventually in the sense
  of the above metric the top 5 database songs are returned with respect to the euclidean
  metric.

REMARK: In both functions the song singal is considered to come from two channels.
        As explained in https://github.com/tyiannak/pyAudioAnalysis/issues/162
        one option is take the channels average. In the code this is implemented by
        the stereo_to_mono function.

"""

import numpy as np
import pickle
import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as mF
from sklearn.decomposition import PCA
from scipy import spatial
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


###############################################################################

def get_similar_manhattan_metric(wav_path, client_id, client_secret, playlist_link):
    
    # download song spotify features
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    playlist_link = playlist_link
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    #track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
    track = sp.playlist_tracks(playlist_URI)["items"][-1] # the song of interest should be in the last playlist position
    track_uri = track["track"]["uri"]
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
    
    # get top 5 similar songs in terms of the manhttan norm
    size = features.shape[0] #size = 171
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i<j:
                matrix[i,j] = spatial.distance.cityblock(features[i],features[j])
                matrix[j,i] = matrix[i,j]
    
    no_of_songs = 5
    index = 0 #because the selected song is in the first position of the similarity matrix
    values = np.sort(matrix[index, :], axis = 0)[1:(no_of_songs+1)]
    indices = matrix[index, :].argsort()[1:(no_of_songs+1)]

    similar_songs = []; similar_scores = []
    for j in range(no_of_songs):
        similar_songs.append(band_song[indices[j]])
        similar_scores.append(round(values[j],2))

    return similar_songs, similar_scores         

###############################################################################

def get_similar_pyaudio_metric(wav_path):
    
    # wav file pyaudio features
    m_win, m_step, s_win, s_step = 5, 5, 1, 1 
    fs, signal = aIO.read_audio_file(os.path.join(wav_path))
    signal = aIO.stereo_to_mono(signal)

    mt, st, feature_names = mF.mid_feature_extraction(signal, fs, 
                                                  m_win * fs, m_step * fs,
                                                  s_win * fs, s_step * fs)
    wav_pyaudio_features = 0
    for j in range(mt.shape[1]):
        wav_pyaudio_features += mt[:,j]
    wav_pyaudio_features = wav_pyaudio_features / mt.shape[1] # this is a np.array of shape (136,)
    
    # database songs pyaudio features
    no_of_songs = 5
    with open("all_features.pickle", "rb") as f: all_features = pickle.load(f)    
    band_song = []
    for j in range(170):
        band_song.append(all_features[j][0] + ' - ' + all_features[j][1])

    pyaudio_features = []
    for j in range(len(all_features)):
        pyaudio_features.append(all_features[j][2][10:]) # since the first 10 features are the spotify features
        # len(pyaudio_features) = 170
        # len(pyaudio_features[0]) = 136
    
    # learn pca via the database songs
    pca = PCA(n_components=2) # choose first two principal components to get visualiation results as well
    ar = np.array(pyaudio_features)
    transformed_database = pca.fit_transform(ar)
    
    # transform the wav file features
    wav_pyaudio_features = np.reshape(wav_pyaudio_features, (1, wav_pyaudio_features.shape[0]))
    transformed_song = pca.transform(wav_pyaudio_features)
    
    # put all transformed 2-dim vectors together
    transformed_all = np.concatenate((transformed_song, transformed_database), axis = 0)

    # compute euclidean similarities
    size = transformed_all.shape[0] #size = 171
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i<j:
                matrix[i,j] = np.linalg.norm(transformed_all[i] - transformed_all[j])
                matrix[j,i] = matrix[i,j]

    # below code computes the values after the smallest one (i.e. 0) and their indices in the 
    index = 0 #because the selected song is in the first position of the similarity matrix
    values = np.sort(matrix[index, :], axis = 0)[1:(no_of_songs+1)]
    indices = matrix[index, :].argsort()[1:(no_of_songs+1)]

    # save the songs and scores in lists
    similar_songs = []; similar_scores = [];
    for j in range(no_of_songs):
        similar_songs.append(band_song[indices[j]])
        similar_scores.append(round(values[j],2))

    return similar_songs, similar_scores