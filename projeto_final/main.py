import importlib
from sklearn.decomposition import PCA
import pandas as pd
from src.features.build_features import DadoFaltantes
from src.features.dimensionamento import dimensionamento
from src.models.train_predict_model import train_predict_model

def main():
    print("Realizando Leitura dos Datasets.")
    market_pop = pd.read_csv('data/raw/estaticos_market.zip')
    portifolio1 = pd.read_csv('data/raw/estaticos_portfolio1.csv', index_col=False)
    #portifolio2 = pd.read_csv('estaticos_portfolio2.csv', index_col=False)
    #portifolio3 = pd.read_csv('estaticos_portfolio3.csv', index_col=False)

    market = market_pop[~market_pop['id'].isin(portifolio1['id'])].reset_index(drop=True)
    portifolio = market_pop[market_pop['id'].isin(portifolio1['id'])].reset_index(drop=True)

    print("Realizando o Preprocessamento.")
    #Tratando o Datasat de treino - Market
    market_exclude_port = DadoFaltantes(market, market_pop).dropnan_coluna_linha()
    market_exclude_port = dimensionamento(market_exclude_port).label_normalizer('normalizer')
    
    df_portifolio = DadoFaltantes(portifolio, market_pop).dropnan_coluna_linha()
    df_portifolio = dimensionamento(df_portifolio).label_normalizer('normalizer')  

    pca_market_nn = PCA(n_components = 6).fit_transform(market_exclude_port.iloc[:, 1:].values)
    pca_portifolio_nn = PCA(n_components =6).fit_transform(df_portifolio.iloc[:, 1:].values)

    print("Realizando o treinamento e previsão.")
    #realiza o treinamento e a predição
    market_exclude_port = market_exclude_port.reset_index(drop=True)
    indice_mais_aderentes = train_predict_model(df_portifolio, pca_market_nn, pca_portifolio_nn)

    print("Salvando o arquivo com dados previstos.")
    #Empresas recomendadas
    empresas_recomendadas = market_exclude_port[market_exclude_port[['id']].index.
                                                    isin(indice_mais_aderentes)].reset_index(drop=True)    
    empresas_recomendadas_toda_info = market[market['id'].
                                                isin(empresas_recomendadas['id'])].reset_index(drop=True)
                                                
    empresas_recomendadas_toda_info = empresas_recomendadas_toda_info.drop(columns=['Unnamed: 0'])
    empresas_recomendadas_toda_info.to_csv('data/processed/empresas_recomendadas.csv', index=False)



if __name__ == "__main__":
    main()
