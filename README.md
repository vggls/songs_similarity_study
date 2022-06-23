# songs_similarity_study (Project for "Multimodal Machine Learning" course)

This repo consists of the following files:

- song_similarity.ipynb: The main Jupyter notebook
   
   Short summary :
   
   Section 1: Spotify features extraction and exploration (10 features per song)
   
   Section 2: PyAudioAnalysis features extraction and exploration (136 features per song)
   
   Section 3: Concatenate Spotify + PyAudio features (146 features per song)
   
   Section 4.1: Use of the 146-dim vector to build similarity matrices based on the euclidean, manhattan, cosine and chebyshev distances and get top 5 recommendations per distance
   
   Section 4.2: Use of the 10 Spotify features to get top 5 recommendations (use of specific features per band/genre)
   
   Section 4.3: Use of the 146-dim vector. We first apply PCA and transfer all database songs to the direction of the first 25 principal components (capture all variance) and then compute the reduced dimension vector distances via the cosine metric to get top 5 recommendations.
   
- presentation.pdf

- article.pdf

- pickle_files folder: The pickle files extracted from the Jupyter notebook

- song_list.txt (the song names comprising the study's database)
 
- demo folder: contains implementation of the pca based metric (see section 4.3 above)
