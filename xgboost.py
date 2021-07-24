from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class XClassifier():  
    def __init__(self, data):
        self.x_test = data['x_test']
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.y_test = data['y_test']


    def train(self, parameters):

        model2 = XGBClassifier(objective='multiclass:softmax', 
                                learning_rate = 0.1,
                                max_depth = 1, 
                                n_estimators = 10)
        model2.fit(self.x_train, self.y_train)
        return model2.predict(self.x_test)

    def eval(self, y_pred):
        return accuracy_score(self.y_test, y_pred, normalize=False)