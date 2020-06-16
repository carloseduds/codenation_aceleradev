import numpy as np
import pandas as pd

class DadoFaltantes:
    def __init__ (self, dataset, df):   
        self.df = df
        self.dataset = dataset
        self.colunas = sorted(set([index for index,
                                   coluna in enumerate(self.nan_contar(self.df)['% Nan']) if (coluna >= 0.25) | (index == 0)]))
    
    def nan_contar(self, df):
        dataset_na = (pd.DataFrame(data=[list(df.columns),
                                list(df.isna().sum()),
                                list(round(df.isna().sum()/df.shape[0], 2))]).
                   T.rename(columns={0:'atributos', 1: "qtde Nan", 2:'% Nan'}))
        return dataset_na

    def dropnan_coluna_linha(self):
        dataset_trat = self._imputar_dados_nan()

        #Exclui as colunas com valores faltantes maiores que 25%
        dataset_trat = dataset_trat.drop(columns=dataset_trat.columns[self.colunas])
        
        #Exclusão de colunas baseadas no conhecimento de negócio do Cientista de Dados
        dataset_trat = dataset_trat.drop(columns=['fl_epp', 'fl_email', 'fl_telefone', 'fl_rm', 'fl_spa', 'fl_antt',
                                                  'fl_veiculo', 'vl_total_veiculos_pesados_grupo',
                                                  'vl_total_veiculos_leves_grupo',
                                                  'de_faixa_faturamento_estimado_grupo',
                                                  'vl_faturamento_estimado_grupo_aux', 
                                                  'vl_faturamento_estimado_aux','dt_situacao', 'idade_emp_cat', 'sg_uf_matriz',
                                                  'fl_me','fl_sa', 'fl_mei', 'fl_ltda', 'fl_st_especial'])
        
        #Exclusão das linhas que apresentam valores igual OUTROS
        pop_dropColumn_outros = dataset_trat[dataset_trat['de_ramo']=='OUTROS'].index
        dataset_trat = dataset_trat.drop(axis=0, labels=pop_dropColumn_outros)
        return dataset_trat
    

    def _imputar_dados_nan(self):
        imputar_nan = ['de_saude_tributaria', 'de_saude_rescencia', 'nu_meses_rescencia',
                       'de_nivel_atividade', 'nm_meso_regiao', 'nm_micro_regiao', 'de_faixa_faturamento_estimado']
        
        criterio_simples = ['EMPRESARIO INDIVIDUAL',
                            'EMPRESA INDIVIDUAL DE RESPONSABILIDADE LIMITADA DE NATUREZA EMPRESARIA',
                            'EMPRESA INDIVIDUAL IMOBILIARIA',
                            'SOCIEDADE SIMPLES LIMITADA',
                            'SOCIEDADE UNIPESSOAL DE ADVOCACIA',
                            'SOCIEDADE SIMPLES PURA',
                            'EMPRESA INDIVIDUAL DE RESPONSABILIDADE LIMITADA DE NATUREZA SIMPLES',
                            'SOCIEDADE SIMPLES EM COMANDITA SIMPLES',
                            'SOCIEDADE EMPRESARIA EM COMANDITA SIMPLES',
                            'SOCIEDADE SIMPLES EM NOME COLETIVO',
                            'SOCIEDADE EMPRESARIA EM NOME COLETIVO']
        
        for columns in imputar_nan:
            #Imputar os dados conforme regras 
            if columns == 'de_saude_tributaria':
                self.dataset[columns] = np.where(self.dataset[columns].isna(),
                                                 'OUTROS',
                                                 self.dataset[columns])

            elif columns == 'de_saude_rescencia':
                self.dataset[columns] = np.where(self.dataset[columns].isna(),
                                                    'SEM INFORMACAO',
                                                    self.dataset[columns])

            elif columns == 'nu_meses_rescencia':
                self.dataset[columns] = np.where(self.dataset['idade_empresa_anos']<1,
                                                    0,
                                                    self.dataset[columns].mean())

            elif columns == 'de_nivel_atividade':
                self.dataset[columns] = self.dataset[columns].fillna('SEM INFORMACAO')
                
            elif columns == 'de_faixa_faturamento_estimado':
                self.dataset[columns] = self.dataset[columns].fillna('SEM INFORMACAO')

            else:
                self.dataset[columns] = self.dataset[columns].fillna("OUTROS")

        self.dataset['fl_optante_simples']= np.where(np.in1d(self.dataset['de_natureza_juridica'], criterio_simples) & 
                                                                (self.dataset['fl_optante_simples'].isna()) &
                                                                (self.dataset['vl_faturamento_estimado_aux']<=480000) &
                                                                (self.dataset['idade_empresa_anos']<5),
                                                                True, False)
        return self.dataset


if __name__ == "__main__":
    DadoFaltantes()
