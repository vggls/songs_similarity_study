# songs_similarity_study

This repo consists of the following files:
- song_similarity.ipynb: The main Jupyter notebook
                          
   Sections 1-3 include the feature extraction and data preprocessing part.
   In order to use the similarity metric functions of section 4, the following pickle files are necessary:
   - subsection 4.1 - "get_similar_standard_metrics" fct - euclidean_similarities.pickle, manhattan_similarities.pickle, cosine_similarities.pickle, chebyshev_similarities.pickle
   - subsection 4.2 - "get_similar_spotify_metric" fct - all_features.pickle
   - subsection 4.3 - "get_similar_pyaudio_metric" fct - all_features.pickle
   
- presentation.pdf: A short presentation of the notebook ideas
- pickle_files folder: The pickle files extracted from the Jupyter notebook
- song_list.txt (the song names comprising the study's database)
- demo folder: py file that contains similarity metrics and omputes top 5 similar database songs for any wav file.
               In the top part of the file a detailed summary of the functions is included.
               Note that the py file needs the all_features.pickle file to run.
