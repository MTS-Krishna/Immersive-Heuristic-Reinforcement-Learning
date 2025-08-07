import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BigDataEnv:
    def __init__(self, file_path, heuristic_fn):
        self.df = pd.read_csv(file_path)
        self.scaler = MinMaxScaler()
        self.df_scaled = self.scaler.fit_transform(self.df.drop(columns=["timestamp"]))
        self.heuristic_fn = heuristic_fn
        self.index = 0

    def reset(self):
        self.index = 0
        return self.df_scaled[self.index] if self.index < len(self.df_scaled) else None

    def step(self, action):
        current = self.df_scaled[self.index]
        original = self.df.iloc[self.index]
        done = False

        expected = self.heuristic_fn(original)
        reward = 1 if action == expected else -1

        self.index += 1
        if self.index >= len(self.df_scaled):
            done = True
            next_state = None
        else:
            next_state = self.df_scaled[self.index]

        return next_state, reward, done
