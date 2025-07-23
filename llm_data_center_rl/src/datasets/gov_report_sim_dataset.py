import os
import random
import pandas as pd


class GovReportSimDataset:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, "../../data/kv_compression_simulate_data/gov_report")
        self.fullkv_df = pd.read_csv(os.path.join(data_dir, "fullkv.csv"))
        self.snapkv_64_df = pd.read_csv(os.path.join(data_dir, "snapkv_64.csv"))
        self.snapkv_128_df = pd.read_csv(os.path.join(data_dir, "snapkv_128.csv"))
        self.snapkv_256_df = pd.read_csv(os.path.join(data_dir, "snapkv_256.csv"))
        self.snapkv_512_df = pd.read_csv(os.path.join(data_dir, "snapkv_512.csv"))
        self.snapkv_1024_df = pd.read_csv(os.path.join(data_dir, "snapkv_1024.csv"))
        self.fullkv = {"times": self.fullkv_df["times"].tolist(),
                       "energies": self.fullkv_df["energies"].tolist(),
                       "scores": self.fullkv_df["scores"].tolist()}
        self.snapkv_64 = {"times": self.snapkv_64_df["times"].tolist(),
                          "energies": self.snapkv_64_df["energies"].tolist(),
                          "scores": self.snapkv_64_df["scores"].tolist()}
        self.snapkv_128 = {"times": self.snapkv_128_df["times"].tolist(),
                           "energies": self.snapkv_128_df["energies"].tolist(),
                           "scores": self.snapkv_128_df["scores"].tolist()}
        self.snapkv_256 = {"times": self.snapkv_256_df["times"].tolist(),
                           "energies": self.snapkv_256_df["energies"].tolist(),
                           "scores": self.snapkv_256_df["scores"].tolist()}
        self.snapkv_512 = {"times": self.snapkv_512_df["times"].tolist(),
                           "energies": self.snapkv_512_df["energies"].tolist(),
                           "scores": self.snapkv_512_df["scores"].tolist()}
        self.snapkv_1024 = {"times": self.snapkv_1024_df["times"].tolist(),
                            "energies": self.snapkv_1024_df["energies"].tolist(),
                            "scores": self.snapkv_1024_df["scores"].tolist()}

    def __len__(self):
        return len(self.fullkv["times"])

    def get_random_item(self, kv_size='fullkv'):
        """
            When take action, random select a request from the simulator dataset based on action (kv_size)
        """
        idx = random.randrange(self.__len__())
        if kv_size == 'fullkv':
            return {"kv_size":  kv_size,
                    "time":     self.fullkv["times"][idx],
                    "energy": self.fullkv["energies"][idx],
                    "score":   self.fullkv["scores"][idx]}
        elif kv_size == 'snapkv_64':
            return {"kv_size":  kv_size,
                    "time":     self.snapkv_64["times"][idx],
                    "energy": self.snapkv_64["energies"][idx],
                    "score":   self.snapkv_64["scores"][idx]}
        elif kv_size == 'snapkv_128':
            return {"kv_size":  kv_size,
                    "time":     self.snapkv_128["times"][idx],
                    "energy": self.snapkv_128["energies"][idx],
                    "score":   self.snapkv_128["scores"][idx]}
        elif kv_size == 'snapkv_256':
            return {"kv_size":  kv_size,
                    "time":     self.snapkv_256["times"][idx],
                    "energy": self.snapkv_256["energies"][idx],
                    "score":   self.snapkv_256["scores"][idx]}
        elif kv_size == 'snapkv_512':
            return {"kv_size":  kv_size,
                    "time":     self.snapkv_512["times"][idx],
                    "energy": self.snapkv_512["energies"][idx],
                    "score":   self.snapkv_512["scores"][idx]}
        elif kv_size == 'snapkv_1024':
            return {"kv_size":  kv_size,
                    "time":     self.snapkv_1024["times"][idx],
                    "energy": self.snapkv_1024["energies"][idx],
                    "score":   self.snapkv_1024["scores"][idx]}
