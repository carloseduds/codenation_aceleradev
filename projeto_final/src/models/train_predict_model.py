from src.features.build_features import DadoFaltantes
from src.features.dimensionamento import dimensionamento
from sklearn.neighbors import NearestNeighbors
from src.evaluate.evaluate import evaluate
import numpy as np
import pandas as pd 

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
            
def empresa_recomendada(predicao, portifolio, market_limpo, market_total, dataset_na):
  #Indices da empresas prevista como mais aderentes
    empresas_recomendadas = portifolio[portifolio.index.
                                                  isin(predicao)].reset_index(drop=True)

    #Empresas recomendadas
    empresas_recomendadas = market_limpo[market_limpo.index.
                                                  isin(predicao)].reset_index(drop=True)

    #Retirando as informaçoes das colunas
    empresas_recomendadas_toda_info = market_total[market_total['id'].
                                                 isin(empresas_recomendadas['id'])].reset_index(drop=True)

    empresas_recomendadas_avaliacao = DadoFaltantes(empresas_recomendadas_toda_info, market_total).dropnan_coluna_linha()
    empresas_recomendadas_avaliacao = dimensionamento(empresas_recomendadas_avaliacao).label_normalizer('normalizer')

    avaliacao = evaluate(portifolio.iloc[:, 1:], empresas_recomendadas_avaliacao.iloc[:, 1:]).cosine_similarity()
    avaliacao = pd.DataFrame(avaliacao[0], columns=['Avaliacao'])

    empresas_recomendadas_avaliacao = pd.concat([empresas_recomendadas_avaliacao, avaliacao], axis=1)
    empresas_recomendadas_avaliacao.to_csv('data/processed/empresas_recomendadas.csv', index=False)
    return empresas_recomendadas_avaliacao