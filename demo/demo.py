# -*- coding: utf-8 -*-
"""
Below code implements the get_similar_pyaudio_metric for any song of interest,
as initially introduced in the the song_similarity.ipynb file.

It takes a wav song as argument and returns the top 5 similar database songs
with respect to the euclidean distance. 
More specifically: The song features are extracted from
the PyAudioAnalysis. At first a PCA transformation is learned on the database songs 
and then the data are transformed in the direction of the first two principal components.
Once the song of interest is selected, in order to compute the distances, its feature
representation is transformed to the learned directions as well.

"""

import numpy as np
import pickle
import os

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as mF

from sklearn.decomposition import PCA

def get_similar_pyaudio_metric(wav_path):
    
    m_win, m_step, s_win, s_step = 5, 5, 1, 1 

    #wav_path = os.path.join(wav_folder_dir, wav)
    fs, signal = aIO.read_audio_file(os.path.join(wav_path))

    # the signal comes from two channels.. 
    # as per **  https://github.com/tyiannak/pyAudioAnalysis/issues/162  ** 
    # one option is to take their average with the stereo_to_mono function
    signal = aIO.stereo_to_mono(signal)

    mt, st, feature_names = mF.mid_feature_extraction(signal, fs, 
                                                  m_win * fs, m_step * fs,
                                                  s_win * fs, s_step * fs)
    wav_pyaudio_features = 0
    for j in range(mt.shape[1]):
        wav_pyaudio_features += mt[:,j]
    wav_pyaudio_features = wav_pyaudio_features / mt.shape[1]
    # this is a np.array of shape (136,)

    no_of_songs = 5
    with open("all_features.pickle", "rb") as f: all_features = pickle.load(f)    
    band_song = []
    for j in range(170):
        band_song.append(all_features[j][0] + ' - ' + all_features[j][1])

    # get pyaudio features, transform them in the direction of the first 3 principal components, 
    # and compute euclidean similarities matrix
    pyaudio_features = []
    for j in range(len(all_features)):
        pyaudio_features.append(all_features[j][2][10:]) # since the first 10 features are the spotify features
        # len(pyaudio_features) = 170
        # len(pyaudio_features[0]) = 136

    pca = PCA(n_components=2) # choose first two principal components to get visualiation results as well
    ar = np.array(pyaudio_features)
    transformed_database = pca.fit_transform(ar)
    print(transformed_database.shape)
    
    wav_pyaudio_features = np.reshape(wav_pyaudio_features, (1, wav_pyaudio_features.shape[0]))
    transformed_song = pca.transform(wav_pyaudio_features)

    transformed_all = np.concatenate((transformed_song, transformed_database), axis = 0)

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