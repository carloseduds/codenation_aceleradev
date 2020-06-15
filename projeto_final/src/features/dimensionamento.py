from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler, StandardScaler

class dimensionamento:
    def __init__(self, dataset):
        self.dataset = dataset
        self.__le = LabelEncoder()
        self.__norm = Normalizer()
        self.__scal = MinMaxScaler()
        self.__stand = StandardScaler()
        
    def __labelEnconder(self):
        for coluna in self.dataset.iloc[:, 1:].select_dtypes(exclude=['float64']).columns:
            self.dataset[coluna] = self.__le.fit_transform(self.dataset[coluna])
        return self.dataset
    
    def __normalizer(self):
        self.dataset.iloc[:, 1:] = self.__norm.fit_transform(self.dataset.iloc[:, 1:])
        return self.dataset
    
    def __scale(self):
        self.dataset.iloc[:, 1:] = self.__scal.fit_transform(self.dataset.iloc[:, 1:])
        return self.dataset
    
    def __stander(self):
        self.dataset.iloc[:, 1:] = self.__stand.fit_transform(self.dataset.iloc[:, 1:])
        return self.dataset
        
    def label_normalizer(self, tipo=None):
        if tipo == 'scale':
            self.dataset = self.__labelEnconder()
            self.dataset = self.__scale()
            return self.dataset
        elif tipo == 'normalizer':
            self.dataset = self.__labelEnconder()
            self.dataset = self.__normalizer()
            return self.dataset
        else:
            self.dataset = self.__labelEnconder()
            self.dataset = self.__stander()
            return self.dataset  