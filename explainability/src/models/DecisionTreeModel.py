from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from explainability.src.Models.Model import Model


class DecisionTreeModel(Model):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def tune(self, X_train, y_train):
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        score = roc_auc_score(y_test, predictions)
        return score