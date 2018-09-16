from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler


class AnnHelper:

    @staticmethod
    def threshold(data, threshold=0.5):
        return data > threshold

    @staticmethod
    def scale_data(x_train, x_test):
        sc = StandardScaler()
        return sc.fit_transform(x_train), sc.transform(x_test)

    @staticmethod
    def create_classifier(optimizer, output_dim, inner_layers):
        data_size = 11
        classifier = Sequential()
        classifier.add(Dense(units=output_dim, kernel_initializer='uniform', activation='relu', input_dim=data_size))
        classifier.add(Dropout(rate=0.1))
        for i in range(inner_layers):
            classifier.add(Dense(units=output_dim, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dropout(rate=0.1))
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return classifier

    @staticmethod
    def create_classifier_a():
        # Initialising the ANN
        classifier = Sequential()
        data_size = 11
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units=data_size, kernel_initializer='uniform', activation='relu', input_dim=data_size))

        for i in range(4):
            # Adding the second hidden layer
            classifier.add(Dense(units=11, kernel_initializer='uniform', activation='relu'))

        # Adding the output layer
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return classifier
