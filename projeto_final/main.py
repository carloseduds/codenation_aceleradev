import importlib
from sklearn.decomposition import PCA
import pandas as pd
from src.features.build_features import DadoFaltantes
from src.features.dimensionamento import dimensionamento
from src.models.train_predict_model import train_predict_model
from src.models.train_predict_model import empresa_recomendada
from src.evaluate.evaluate import evaluate


def main():
    print("Realizando Leitura dos Datasets.")
    market_pop = pd.read_csv('data/raw/estaticos_market.zip')
    portifolio1 = pd.read_csv('data/raw/estaticos_portfolio1.csv', index_col=False)
    portifolio2 = pd.read_csv('data/raw/estaticos_portfolio2.csv', index_col=False)
    portifolio3 = pd.read_csv('data/raw/estaticos_portfolio3.csv', index_col=False)

    n=1
    for portifolio_file in [portifolio1, portifolio2, portifolio3]:
        market = market_pop[~market_pop['id'].isin(portifolio_file['id'])].reset_index(drop=True)
        portifolio = market_pop[market_pop['id'].isin(portifolio_file['id'])].reset_index(drop=True)

        print("Realizando o Preprocessamento.")
        #Tratando o Datasat de treino - Market
        #Cria o dataset com a quantiade faltas
        dataset_na = DadoFaltantes(market_pop, market_pop).nan_contar(market_pop)

        #realiza o preprocessamento tanto do dataset market quanto o portifolio
        market_exclude_port = DadoFaltantes(market, market_pop).dropnan_coluna_linha()
        market_exclude_port = dimensionamento(market_exclude_port).label_normalizer('normalizer')
        
        df_portifolio = DadoFaltantes(portifolio, market_pop).dropnan_coluna_linha()
        df_portifolio = dimensionamento(df_portifolio).label_normalizer('normalizer')  

        #Aplica PCA nos dataset
        pca_market_nn = PCA(n_components = 6).fit_transform(market_exclude_port.iloc[:, 1:].values)
        pca_portifolio_nn = PCA(n_components =6).fit_transform(df_portifolio.iloc[:, 1:].values)

        print("Realizando o treinamento e previsão.")
        #realiza o treinamento e a predição
        indice_mais_aderentes = train_predict_model(df_portifolio, pca_market_nn, pca_portifolio_nn)

        print("Salvando o arquivo com dados previstos.")
        #Empresas recomendadas
        dataset_recomenda = empresa_recomendada(indice_mais_aderentes, df_portifolio, market_exclude_port, market_pop, dataset_na)
        print(f'A média de similaridade de cossine para o portifolio{n} é de: {dataset_recomenda["Avaliacao"].mean()}')
        n +=1

if __name__ == "__main__":
    main()
