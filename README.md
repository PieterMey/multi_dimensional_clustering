MD_clustering is a package that allows for exploratory analysis of multi-dimensional data through KMeans clustering provided by scikit-learn.

DEPENDENCIES:
- python =3.8.3
- numpy = 1.18.5
- matplotlib = 3.4.2
- plotly = 5.1.0
- scikit-learn = 0.24.2 
- seaborn = 0.11.1

The package operates as a class. It has implementations to handle preprocessed/unpreprocessed data. It has the build in ability to display the elbow-method to determine optimal number of clusters. As well as the loading scores involved for n PCA components. The class can display 1-D, 2-D, 3-D visualizations based on the output of the KMeans algorithm and the top 3 PCA components. Finally, if desired it can also display a pairwise plot of all features. 

Below a list of possible functionality is shown, a deeper explenation is given when typing help(MD_clustering()):
- from multi_dimensional_clustering import MD_clustering
- MDC = MD_clustering() # Creating the object
- MDC.load_data(PATH, label_column_name=COL_NAME, preprocessed=BOOL) # Loading the data, if there is a label column that should be appended or wants to be saved please specify. Also if the data has already been preprocessed.
- MDC.drop_rows([int(s)]) # drop specific rows
- MDC.drop_cols(cols=[COL_NAMES], save_cols=[COL_NAMES]) # drop unwanted cols, drop wanted columns that should be appended when saving data.
- MDC.scale_data(scaler=SCALER) # Scale data by SCALER if not preprocessed yet.
- MDC.get_loading_scores() # Display loading scores for n pca components.
- MDC.get_n_clusters() # Display elbow-method graph through analyzing intertia from KMeans
- MDC.cluster(clusters_n=int) # Cluster data through KMeans on n clusters
- MDC.visualize('3D') # Visualize results in 3D plot 
- MDC.inverse_scale() # Inverse scale the data to original format if so desired
- MDC.concat_saved_cols() # concat the dropped wanted columns 
- MDC.save_data() # Save data, possible file path can be given
- MDC.pairwise_plot() # Plot a pairwise plot for further analysis.
