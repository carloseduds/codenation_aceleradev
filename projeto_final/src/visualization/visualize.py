import matplotlib.pyplot as plt
import seaborn as sns


def sns_plotv(x, y, data,  titulo, xlabel):
    plt.figure(figsize=(12, 8))
    graph = sns.barplot(x, y, palette="rocket", data=data)
    graph.axes.set_title(titulo,fontsize=20)
    graph.set_xlabel(xlabel,fontsize=18)
    graph.set_ylabel('Total de Empresas',fontsize=18)
    graph.tick_params(labelsize=16)
    sns.despine(left=True)
    plt.show()

def sns_ploth(x, y, data,  titulo, ylabel):
    plt.figure(figsize=(12, 8))
    graph = sns.barplot(x, y, palette="rocket", data=data)
    graph.axes.set_title(titulo,fontsize=20)
    graph.set_xlabel('Total de Empresas',fontsize=18)
    graph.set_ylabel(ylabel,fontsize=14)
    graph.tick_params(labelsize=10)
    sns.despine(left=True)
    plt.show()