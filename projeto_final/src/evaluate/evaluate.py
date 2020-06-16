from scipy.spatial import distance
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cdist


class evaluate:
    def __init__(self, portifolio, predicao):
        self.portifolio = portifolio
        self.predicao = predicao
        
    def cosine_similarity(self):
        for self.portifolio in self.predicao.values:
            similarities = 1 - cdist([self.portifolio], self.predicao.dropna().values, metric='cosine')
        return similarities
    
    def euclidian(self):
        port = np.ravel(self.portifolio)
        pred = np.ravel(self.predicao.iloc[0: self.portifolio.shape[0]]) 
        dst = distance.euclidean(port, pred)
        return dst
        
    def mannhatan(self):
        port = np.ravel(self.portifolio)
        pred = np.ravel(self.predicao.iloc[0: self.portifolio.shape[0]])     
        dst = distance.cityblock(port, pred)
        return dst