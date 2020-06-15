from sklearn.neighbors import NearestNeighbors
import numpy as np

def train_predict_model(dataset_portfolio, dataset_market, dataset_previsao):
    quantidade_empresas = 0
    for k in range(1,50):
        if quantidade_empresas < dataset_portfolio.shape[0]:
            knn =  NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
            predicao = knn.fit(dataset_market)    
            predicao_v1 = predicao.kneighbors(dataset_previsao)
            quantidade_empresas = len(set(np.ravel(predicao_v1[1])))
    #Indices da empresas prevista como mais aderentes
    indice_mais_aderentes = list(set(np.ravel(predicao_v1[1])))
    return indice_mais_aderentes