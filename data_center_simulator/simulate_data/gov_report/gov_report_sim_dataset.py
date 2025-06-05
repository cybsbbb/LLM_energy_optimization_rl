from data_center_simulator.simulate_data.gov_report.sim_scores import *
from data_center_simulator.simulate_data.gov_report.sim_times import *
from data_center_simulator.simulate_data.gov_report.sim_energies import *
import random

class GovReportSimDataset:
    def __init__(self):
        self.fullkv = {"times": fullkv_times, "energies": fullkv_energies, "scores": fullkv_scores}
        self.snapkv_64 = {"times": snapkv_64_times, "energies": snapkv_64_energies, "scores": snapkv_64_scores}
        self.snapkv_128 = {"times": snapkv_128_times, "energies": snapkv_128_energies, "scores": snapkv_128_scores}
        self.snapkv_256 = {"times": snapkv_256_times, "energies": snapkv_256_energies, "scores": snapkv_256_scores}
        self.snapkv_512 = {"times": snapkv_512_times, "energies": snapkv_512_energies, "scores": snapkv_512_scores}
        self.snapkv_1024 = {"times": snapkv_1024_times, "energies": snapkv_1024_energies, "scores": snapkv_1024_scores}

    def __len__(self):
        return len(self.fullkv["times"])

    def get_random_item(self, kv_size='fullkv'):
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


if __name__ == "__main__":
    gov_report_sim_data = GovReportSimDataset()
    for i in range(100):
        print(gov_report_sim_data.get_random_item(kv_size="fullkv"))
        print(gov_report_sim_data.get_random_item(kv_size="snapkv_64"))
        print(gov_report_sim_data.get_random_item(kv_size="snapkv_128"))
        print(gov_report_sim_data.get_random_item(kv_size="snapkv_256"))
        print(gov_report_sim_data.get_random_item(kv_size="snapkv_512"))
        print(gov_report_sim_data.get_random_item(kv_size="snapkv_1024"))

