import pandas


class DataReader:
    def __init__(self, file_name):
        self._file_name = file_name

    def read(self):
        dataset = pandas.read_csv(self._file_name)
        X = dataset.iloc[:, 3:13].values
        y = dataset.iloc[:, 13].values
        return X, y
