import pandas as pd
import numpy as np


class ImportCSV:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def read_csv(self):
        data = pd.read_csv(self.csv_path)
        return data
