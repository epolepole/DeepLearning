from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier

from modules.DataReader import DataReader
from modules.ann_helper import AnnHelper


def encode_labels(ix):
    ix[:, 1] = LabelEncoder().fit_transform(ix[:, 1])
    ix[:, 2] = LabelEncoder().fit_transform(ix[:, 2])
    ix = OneHotEncoder(categorical_features=[1]).fit_transform(ix).toarray()
    return ix[:, 1:]


if __name__ == '__main__':
    file_name = 'data/Churn_Modelling.csv'
    x, y = DataReader(file_name).read()
    x = encode_labels(x)
    data_size = 1
    x_data = x[:data_size, :]
    x_known = x[data_size:, :]
    y_known = y[data_size:]
    x_train, x_test, y_train, y_test = train_test_split(x_known, y_known, test_size=0.2, random_state=0)
    x_train, x_test = AnnHelper.scale_data(x_train, x_test)

    # Normal ANN
    # classifier = AnnHelper.create_classifier('adam', 6, 1)
    # classifier.fit(x_train, y_train, batch_size=10, epochs=100)
    # y_pred = AnnHelper.threshold(classifier.predict(x_test))
    # cm = confusion_matrix(y_test, y_pred)
    # y_result = AnnHelper.threshold(classifier.predict(x_data))

    # With Cross Val Score. Ensure low variance
    classifier = KerasClassifier(build_fn=AnnHelper.create_classifier, batch_size=10, epochs=100, optimizer='adam', output_dim=6, inner_layers=1)
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print('mean: {}, variance: {}'.format(str(mean), str(variance)))

    # With GridSearchCV -> Parameters optimization
    # classifier = KerasClassifier(build_fn=AnnHelper.create_classifier)
    #
    # parameters = {
    #     'batch_size':   [25, 32],
    #     'epochs':       [100, 500],
    #     'optimizer':    ['adam', 'rmsprop'],
    #     'output_dim':   [6, 8, 11],
    #     'inner_layers': [1, 2, 6]
    # }
    # grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
    # grid_search_fitted = grid_search.fit(x_train, y_train, verbose=1)
    #
    # best_parameters = grid_search_fitted.best_params_
    # best_accuracy = grid_search_fitted.best_score_
