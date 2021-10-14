import random
import glob

import os

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import plotly as py

import plotly.graph_objs as go
from plotly.offline import iplot

from IPython.display import display
from IPython.display import Image

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

class MD_clustering:

    def __init__(self):
        self.colors = ['rgba(255, 128, 255, 0.8)','rgba(255, 128, 2, 0.8)','rgba(0, 255, 200, 0.8)','rgba(0, 0, 255, 0.8)','rgba(255, 255,115, 0.8)','rgba(255, 0,0, 0.8)','rgba(98, 0,131, 0.8)', 'rgba(255,235,215,0.8)', 'rgba(125,38,205,0.8)','rgba(0, 0, 0, 0.8)']
        self.label_col_bool = False
        self.label_col = None
        self.plotX = None
        self.data_clustered = None
        self.data_scaled = None
        self.data_inverse_scaled = None
        self.save_col = None
        self.data_final = None
        self.loadings = None

    def load_data(self, filename='', sep=',',decimal='.',label_column_name=None, preprocessed = False):
        """
        Function loads data from a csv file only based on filename/path.

        Args:
            filename (str, non-optional): Origin filename/path.
            sep (str, optional): Seperator for csv file. Defaults to ','.
            decimal (str, optional): Decimal for csv file. Defaults to '.'.
            label_column_name (str, optional): If data set contains a label column (i.e., company names). Defaults to None.
            preprocessed (bool, optional): If data has been preprocessed prior then it will save it to according variables. Defaults to False.
        """
        if filename == '':
            print('File path for loading data is empty.')
        else:
            self.data = pd.read_csv(filepath_or_buffer=filename, sep=sep, decimal=decimal)

            if label_column_name != None:
                self.label_column_name = label_column_name
                self.label_col = pd.DataFrame(self.data[self.label_column_name])
                self.data.drop(self.label_column_name, inplace=True, axis=1)
                self.label_col_bool = True

            if preprocessed:
                self.data_scaled = self.data
    
    def save_data(self, filename = 'final_result_MD.csv', X = '', sep = ';', decimal='.'):
        """
        Function saves the final resulting dataframe into a csv file within the current folder.
        Args:
            filename (str, optional): Output filename. Defaults to 'final_result_MD.csv'.
            X (str, optional): [description]. Defaults to self.data_clustered / self.data_final.
            sep (str, optional): Seperator for csv file. Defaults to ';'.
            decimal (str, optional): Decimal for csv file. Defaults to '.'.
        """
            
        if not(isinstance(X, pd.DataFrame)) and X == '' and not(isinstance(self.data_final, pd.DataFrame)):
            X = self.data_clustered
        elif isinstance(self.data_final, pd.DataFrame):
            X = self.data_final
        
        if not os.path.exists('output_data'):
            os.makedirs('output_data')
            print('Created new folder @output_data where all results will be stored.')
        X.to_csv('output_data/'+filename, sep=sep, decimal=decimal)

        

    def scale_data(self, X='', scaler=MinMaxScaler()):
        """
        Function scales data 'X' from user inputed desired 'scaler'. (i.e., MaxAbsScaler, MinMaxScaler, StandardScaler)
        
        Args:
            X (pd.DataFrame, optional): A data set given as a dataframe to be scaled. Defaults to self.data. 
            scaler (function, optional): A scaling function from sklearn, or one that has the functionality of 'fit_transform' and 'inverse_transform'. Defaults to MinMaxScaler.
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':
            X = self.data.copy()
        self.scaler = scaler
        X = pd.DataFrame(self.scaler.fit_transform(X), columns = X.columns)
        self.data_scaled = X

    def inverse_scale(self, X=''):
        if not(isinstance(X, pd.DataFrame)) and X == '':
            X = self.data_scaled.copy()
        self.data_inverse_scaled = pd.DataFrame(self.scaler.inverse_transform(X), columns= X.columns)
        self.data_inverse_scaled = pd.concat([self.label_col, self.data_inverse_scaled], axis=1)

        if 'Cluster' not in self.data_inverse_scaled.columns:
            self.data_inverse_scaled = pd.concat([self.clusters_l, self.data_inverse_scaled], axis=1)


    def drop_cols(self, cols='',save_cols=''):
        """
        Function drops an entire column from loaded self.data.
        
        Args:
            cols (list, optional): A list of column names to be dropped.
            save_cols (list, optional): A list of column names to be dropped but will be saved for the possibility of later re-attachment. 
            All operations such as scaling will be perfomed on this DF too.
        """
        if isinstance(self.data_scaled, pd.DataFrame):
            if isinstance(cols, list):
                self.dropped_columns = self.data[cols]
                self.data_scaled.drop(cols, inplace=True, axis=1)
            elif cols != '':
                self.dropped_columns = self.data[[cols]]
                self.data_scaled.drop([cols], inplace=True, axis=1)
            
            if isinstance(save_cols, list):
                self.save_col = self.data_scaled[save_cols]
                self.data_scaled.drop(save_cols, inplace=True, axis=1)
            elif save_cols != '':
                self.save_col = self.data_scaled[save_cols]
                self.data_scaled.drop([save_cols], inplace=True, axis=1)
        elif isinstance(self.data, pd.DataFrame):
            if isinstance(cols, list):
                self.dropped_columns = self.data[cols]
                self.data.drop(cols, inplace=True, axis=1)
            elif cols != '':
                self.dropped_columns = self.data[[cols]]
                self.data.drop([cols], inplace=True, axis=1)
            
            if isinstance(save_cols, list):
                self.save_col = self.data[save_cols]
                self.data.drop(save_cols, inplace=True, axis=1)
            elif save_cols != '':
                self.save_col = self.data[save_cols]
                self.data.drop([save_cols], inplace=True, axis=1)
    
    def concat_saved_cols(self, X=''):
        if not(isinstance(X, pd.DataFrame)) and X == '':
            if isinstance(self.data_inverse_scaled, pd.DataFrame):
                X = self.data_inverse_scaled.copy()
            else:
                X = self.data_clustered.copy()
            self.data_final = pd.concat([X, self.save_col], axis =1)

    def drop_rows(self, rows):
        """
        Function drops [row_index_1, row_index_n, ...] from loaded data set in self.data and also from label_col if it exists.
        Make sure rows is a list.
       
        Args:
            rows (list, non-optional): A list of row indices.
        """
        self.data.drop(rows, axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        if isinstance(self.label_col, pd.DataFrame):
            self.label_col.drop(list(rows), inplace=True, axis=0)
            self.label_col.reset_index(drop=True, inplace=True)

        if isinstance(self.save_col, pd.DataFrame):
            self.save_col.drop(list(rows), inplace=True, axis=0)
            self.save_col.reset_index(drop=True, inplace=True)

    def get_n_clusters(self, X='', min_n = 1, max_n = 11, show=True):
        """
        This function visualizes the 'Elbow Method' for KMeans, showing the 
        predicted optimal n_clusters for 'X' based on inertia.
        
        Args:
            X (pd.DataFrame, non-optional): Unclustered original data set.
            min_n (int, optional): Minimum desired clusters. Defaults to 1.
            max_n (int, optional): Maximum desired clusters. Defaults to 11.
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':
            if isinstance(self.data_scaled, pd.DataFrame):
                X = self.data_scaled.copy()
            elif isinstance(self.data, pd.DataFrame):
                X = self.data.copy()

        pca = PCA(n_components=np.shape(X)[1])
        principalComponents = pca.fit_transform(X)
        PCA_components = pd.DataFrame(principalComponents)    
        ks = range(min_n,max_n)
        inertias = []

        for k in ks:
            # Create a KMeans instance with k clusters: model
            km_model = KMeans(n_clusters=k)

            # Fit model to samples
            km_model.fit(PCA_components.iloc[:,:3])

            # Append the inertia to the list of inertias
            inertias.append(km_model.inertia_)
        if show:
            plt.plot(ks, inertias, '-o', color='black')
            plt.xlabel('number of clusters, k')
            plt.ylabel('inertia')
            plt.xticks(ks)
            plt.show()
        

    def get_loading_scores(self, X='', pca_components=3, positive_cor = False, show=True):
        """
        The function visualizes the importance of each feature in data set 'X' to n 'pca_components'.
        
        Args:
            X (pd.DataFrame, non-optional): Original unclustered data set.
            pca_components (int, non-optional): Desired amount of visualized principal components. Defaults to 3.
            positive_cor (bool, optional): If user only wants to see the positive correlated features to each PC. Defaults to False.
            show (bool, optional): If user wants to see the graphical displays. Defaults to True
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':
            if isinstance(self.data_scaled, pd.DataFrame):
                X = self.data_scaled.copy()
            elif isinstance(self.data, pd.DataFrame):
                X = self.data.copy()

        # Train PCA
        pca = PCA().fit(X)

        if show:
            # Cumulative Variance Plot
            plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
            plt.title('Cumulative explained variance by number of principal components', size=20)
            plt.show()

        # PCA Loading Score Plots
        self.loadings = pd.DataFrame(data=pca.components_.T * np.sqrt(pca.explained_variance_), columns=[f'PC{i}' for i in range(1, len(X.columns) + 1)], index = X.columns)
        self.pca_loadings_list = []
        for i in range(1, pca_components+1):
            pc_loadings = self.loadings.sort_values(by='PC{}'.format(i), ascending=False)[['PC{}'.format(i)]]
            pc_loadings = pc_loadings.reset_index()
            pc_loadings.columns = ['Attribute', 'CorrelationWithPC{}'.format(i)]
            self.pca_loadings_list.append(pc_loadings)

            # Get positive correlation, remove if want to see all correlations to a PC
            if positive_cor:
                pc_loadings = pc_loadings[pc_loadings['CorrelationWithPC{}'.format(i)] >= 0]

            if show:
                plt.bar(x=pc_loadings['Attribute'], height=pc_loadings['CorrelationWithPC{}'.format(i)], color='#087E8B')
                plt.title('PCA loading scores (principal component #{}'.format(i), size=20)
                plt.xticks(rotation='vertical')
                plt.show()

                # Display values as a dataframe
                display(pd.DataFrame(pc_loadings))
        
    def cluster(self, X='', clusters_n=3,seed=None):
        """
        Function clusters the data 'X' into 'clusters_n' clusters'. It then also finds the 1st, 2nd and 3rd PC.

        Args:
            X (pd.DataFrame, non-optional): A dataframe of X unclustered data.
            clusters_n (int, non-optional): An integer to cluster the data 'X' into n clusters. Defaults to 3.
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':
            if isinstance(self.data_scaled, pd.DataFrame):
                X = self.data_scaled.copy()
            elif isinstance(self.data, pd.DataFrame):
                X = self.data.copy()

        x = X.copy()
        self.n_clusters = clusters_n

        self.kmeans = KMeans(n_clusters=clusters_n, random_state=seed)
        self.kmeans.fit(x)
        self.clusters_l = pd.DataFrame(self.kmeans.predict(x), columns=['Cluster'])
        x = pd.concat([self.clusters_l, x],axis=1)
        self.get_PCA(x)
        if self.label_col_bool: x = pd.concat([self.label_col,x],axis=1)
        self.data_clustered = x

    def get_PCA(self, X=''):
        """
        Function gets the PCA with 1, 2 and 3 principal components from clustered data; these are needed for plotting in visualize. 
        See following link for more information on PCA: https://builtin.com/data-science/step-step-explanation-principal-component-analysis

        Args:
            X (pd.DataFrame): A clustered pandas DataFrame. Must include a column with 'Cluster' labels.
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':
            if isinstance(self.data_clustered, pd.DataFrame):
                X = self.data_clustered.copy()
            elif isinstance(self.data, pd.DataFrame):
                X = self.data.copy()
                
        plotX = X.copy()
        plotX.columns = X.columns

        pca_1d = PCA(n_components=1) # PCA with one principal components
        pca_2d = PCA(n_components=2) # PCA with two principal components
        pca_3d = PCA(n_components=3) # PCA with three principal components

        # We build our new DataFrames:
        PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(['Cluster'], axis=1))) # This DataFrame holds 1 principal component
        PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(['Cluster'], axis=1))) # This DataFrame holds 2 principal component
        PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(['Cluster'], axis=1))) # This DataFrame holds 3 principal component

        # Rename the columns of these newly created DataFrames:
        PCs_1d.columns = ['PC1_1d']
        PCs_2d.columns = ['PC1_2d', 'PC2_2d']
        PCs_3d.columns = ['PC1_3d', 'PC2_3d', 'PC3_3d']

        # WE concatenate these newly created DataFranes to plotX so that they can be used by plotX as columns.
        plotX = pd.concat([plotX, PCs_1d, PCs_2d, PCs_3d], axis=1, join='inner')

        # and we create one new column for plotX so that we can use it for 1-D visualization
        plotX['dummy'] = 0

        self.plotX = plotX
 
    def visualize(self, graph='2D', X='', show_loading_scores=False, show_mesh=False):
        """
        The function is able to visualize clustered data 'X' in different dimensions 1D || 2D || 3D projected from prinicipal components of the data set features.

        Args:
            X (pd.DataFrame): A clustered DataFrame. 
            graph (str, optional): A string '1D', '2D' or '3D' for the desired graph representation. Defaults to 2D.                                                                   
            show_loading_scores (bool, optional): If user wants the loading scores desplayed on the 2D and 3D graph. Defaults to False.
            show_mesh (bool, optional): If the user wants a vague representation of the cluster boundary in the 3D space. Defaults to False.
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':                       
            X = self.data_clustered.copy()       
                     
        if not(isinstance(self.data_clustered, pd.DataFrame)):
            print('''Data has not been clustered, either use 'cluster()' or 'set_cluster_data()'. \n 
            Data will now be clustered with n_clusters = 4.''')
            self.cluster(X, 4)
            
        if not(isinstance(self.plotX, pd.DataFrame)):
            self.get_PCA(X)
        
        if not(isinstance(self.loadings, pd.DataFrame)):
            self.get_loading_scores(show=False)

        clusters = []

        if isinstance(self.data_scaled, pd.DataFrame):
            features = self.data_scaled.columns
        elif isinstance(self.data, pd.DataFrame):
            features = self.data.columns

        PLOT = go.Figure()

        for i in range(self.n_clusters):
            clusters.append(self.plotX[self.plotX['Cluster'] == i])

            if self.label_col_bool:
                names = self.data_clustered[self.data_clustered['Cluster']==i]
                names = names[self.label_column_name]
            else:
                names = None

            if graph == '3D':
                title = "Visualizing Clusters in Three Dimensions Using PCA"
                PLOT.add_trace(go.Scatter3d(
                        x = clusters[i]["PC1_3d"],
                        y = clusters[i]["PC2_3d"],
                        z = clusters[i]["PC3_3d"],
                        mode = "markers",
                        name = "Cluster {}".format(i),
                        marker = dict(color = self.colors[i]),
                        text = names))
                if show_mesh:
                    PLOT.add_trace(go.Mesh3d(
                        alphahull = 7,
                        name = "y",
                        opacity = 0.1,
                        x = clusters[i]["PC1_3d"],
                        y = clusters[i]["PC2_3d"],
                        z = clusters[i]["PC3_3d"]
                    ))
                PLOT.update_layout(width=1600, height=800, autosize=True, showlegend=True,
                            scene=dict(xaxis=dict(title='PC1',ticklen= 5, zeroline= False), 
                                    yaxis=dict(title='PC2',ticklen=5,zeroline= False), 
                                    zaxis=dict(title='PC3',ticklen=5,zeroline= False)))
                if show_loading_scores and i == self.n_clusters-1:
                    for j, feature in enumerate(features):
                        PLOT.add_trace(go.Scatter3d(
                                x = [0,list(self.loadings.iloc[j][['PC1']])[0]],
                                y = [0,list(self.loadings.iloc[j][['PC2']])[0]],
                                z = [0,list(self.loadings.iloc[j][['PC3']])[0]],
                                name=feature,
                                text=feature,
                                marker = dict(
                                    size=2,
                                    color=self.colors[-1],
                                ),
                                line = dict(
                                    color=self.colors[-1],
                                    width=10
                                )
                        ))

            elif graph == '2D':
                PLOT.add_trace(go.Scatter(
                        x = clusters[i]["PC1_2d"],
                        y = clusters[i]["PC2_2d"],
                        mode = "markers",
                        name = "Cluster {}".format(i),
                        marker = dict(color = self.colors[i]),
                        text = names))
                title = "Visualizing Clusters in Two Dimensions Using PCA"
                PLOT.update_layout(dict(title = title,
                            xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                            yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                            ))
                if show_loading_scores and i == self.n_clusters-1:
                    for j, feature in enumerate(features):
                        PLOT.add_shape(
                            type='line',
                            x0=0, y0=0,
                            x1=self.loadings.iloc[j][0],
                            y1=self.loadings.iloc[j][1],
                        )
                        PLOT.add_annotation(
                            x=self.loadings.iloc[j][0],
                            y=self.loadings.iloc[j][1],
                            ax=0,ay=0,
                            xanchor='center',
                            yanchor='bottom',
                            text=feature
                        )

            elif graph == '1D':
                PLOT.add_trace(go.Scatter(
                        x = clusters[i]["PC1_1d"],
                        y = clusters[i]["dummy"],
                        mode = "markers",
                        name = "Cluster {}".format(i),
                        marker = dict(color = self.colors[i]),
                        text = names))
                title = "Visualizing Clusters in One Dimension Using PCA"
                PLOT.update_layout(dict(title = title,
                            xaxis= dict(title= 'PC1', ticklen= 5, zeroline= False), 
                            ))

        iplot(PLOT)

    def pairwise_plot(self, X='', save_img = False, img_name = 'pairplot.png'):
        """
        The function creates a seaborn pairwise plot. Plotting the clustering results against each feature combination.
        Args:
            X (pd.DataFrame, optional): A clustered dataframe. Defaults to self.data_clustered.
            save_img (bool, optional): Boolean if the image should be saved within the current file. Defaults to False.
            img_name (str, optional): The desired file name for the image, should end in '.png'. Defaults to 'pairplot.png'.
        """
        if not(isinstance(X, pd.DataFrame)) and X == '':
            X = self.data_clustered.copy()

        sns_plot = sns.pairplot(X, hue = 'Cluster')
        plt.show()

        if save_img:
            plt.clf()
            img_name = str(random.randint(0,1000))+'_'+img_name
            sns_plot.savefig(img_name)
            filename = glob.glob('./'+img_name)[0]
            Image(filename=filename)

    def set_label_column_name(self, label_column_name):
        self.label_column_name = label_column_name
    
    def set_data(self, x):
        if isinstance(x, pd.DataFrame):
            self.data = x
        else:
            self.data = pd.DataFrame(x)
    
    def set_data_scaled(self, x):
        if isinstance(x, pd.DataFrame):
            self.data_scaled = x
        else:
           self.data_scaled = pd.DataFrame(x)

    def set_data_clustered(self, x):
        if isinstance(x, pd.DataFrame):
            self.data_clustered = x
        else: 
           self.data = pd.DataFrame(x)
        
