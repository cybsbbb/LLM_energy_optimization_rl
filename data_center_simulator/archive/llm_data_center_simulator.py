import time
from heapq import heappush, heappop
from data_center_simulator.simulate_data.gov_report.gov_report_sim_dataset import GovReportSimDataset
from data_center_simulator.energy_price.caiso import data_20250421
from data_center_simulator.utils import setup_random, generate_bernoulli


class LLMDataCenterSimulator:
    def __init__(self, agent):
        self.gov_report_sim_dataset = GovReportSimDataset()
        # one day sim
        self.total_time = 24 * 3600 * 1000
        # simulator time interval
        self.time_interval = 10
        # Bernoulli prob
        self.bernoulli_prob = 0.2
        # number of server
        self.server_num = 200
        # max wait time (otherwise, user will leave) (10 second)
        self.max_wait_time = 10000
        # energy price (every 5 minutes, 300*1000 ms)
        self.energy_price = data_20250421
        self.five_minutes_ms = 300 * 1000
        # energy unit change
        self.j2mwh = 3_600_000_000
        # PUE (Power usage effectiveness)
        self.pue = 1.3
        # agent
        self.agent = agent
        # init
        self.init_simulator()

    def init_simulator(self):
        # time stamp queue (used for this simulator)
        self.event_queue = []
        # processing queue
        self.processing_server_num = 0
        # waiting queue
        self.waiting_queue = []
        # summary related
        self.success_request_num = 0
        self.failed_request_num = 0
        self.latency_list = []
        self.score_list = []
        self.tot_energy = 0
        self.tot_energy_cost = 0

    def get_cur_energy_price(self, cur_time):
        return self.energy_price[cur_time//self.five_minutes_ms]

    def handle_failed_request(self, cur_time):
        while len(self.waiting_queue) > 0 and self.waiting_queue[0][0] < cur_time - self.max_wait_time:
            self.waiting_queue.pop(0)
            self.failed_request_num += 1

    def handle_new_request(self, cur_time, new_request):
        # There is available server exist
        if self.processing_server_num < self.server_num:
            new_request_time = round(new_request["time"] * 1000)
            finish_time = cur_time + new_request_time
            self.processing_server_num += 1
            heappush(self.event_queue, (finish_time, time.time_ns(), {"type": "finish", "request": new_request}))
        else:
            heappush(self.waiting_queue, (cur_time, {"type": "waiting", "request": new_request}))

    def handle_timestamp(self, cur_time, cur_event):
        self.handle_failed_request(cur_time)
        heappush(self.event_queue, (cur_time + self.time_interval, time.time_ns(), cur_event))
        # Bernoulli to determine if there is a new request
        flag = generate_bernoulli(self.bernoulli_prob)
        if flag:
            state = {"processing_server_num": self.processing_server_num, "total_server_num": self.server_num,
                     "waiting_request_num": len(self.waiting_queue), "energy_price": self.get_cur_energy_price(cur_time),
                     "request_priority": 1}
            pre_time = round(time.time() * 1000)
            action = self.agent(state)
            after_time = round(time.time() * 1000)
            cur_time += after_time - pre_time
            new_request = self.gov_report_sim_dataset.get_random_item(action)
            new_request["energy_price"] = self.get_cur_energy_price(cur_time)
            self.handle_new_request(cur_time, new_request)

    def handle_finish(self, cur_time, cur_event):
        self.handle_failed_request(cur_time)
        self.success_request_num += 1
        self.processing_server_num -= 1
        self.score_list.append(cur_event["request"]["score"])
        self.tot_energy += cur_event["request"]["energy"]
        self.tot_energy_cost += cur_event["request"]["energy"] * cur_event["request"]["energy_price"]
        if len(self.waiting_queue) > 0:
            # currently pick the longest waited request (should have better strategy)
            oldest_time, oldest_event = heappop(self.waiting_queue)
            latency = cur_time - oldest_time
            self.latency_list.append(latency)
            self.handle_new_request(cur_time, oldest_event["request"])


    def run_simulator(self):
        setup_random()
        self.init_simulator()
        heappush(self.event_queue, (0, time.time_ns(), {"type": "timestamp"}))
        while True:
            cur_time, _, cur_event = heappop(self.event_queue)
            if cur_time >= self.total_time:
                break
            if cur_event["type"] == "timestamp":
                self.handle_timestamp(cur_time, cur_event)
            elif cur_event["type"] == "finish":
                self.handle_finish(cur_time, cur_event)

        self.print_summary()
        print("Simulation finished")
        return

    def print_summary(self):
        print("Summary: ")
        print(f"Total Success Requests: {self.success_request_num}")
        print(f"Total Failed Requests: {self.failed_request_num}")
        print(f"Success Rate: {self.success_request_num / (self.success_request_num + self.failed_request_num)}")
        if len(self.latency_list) > 0:
            print(f"Average Latency: {(sum(self.latency_list) / len(self.latency_list)) / 1000} s")
        else:
            print(f"Average Latency: {0} ms")
        print(f"Average Score: {(sum(self.score_list) / len(self.score_list)) * 100}")
        print(f"Total Energy: {self.tot_energy} J")
        print(f"Total Energy Cost: {self.tot_energy_cost * self.pue / self.j2mwh}")
        print(f"Total profit: {self.success_request_num / 1000 * 0.02}")


if __name__ == "__main__":
    from archive.simple_agent import all_fullkv_agent

    simulator = LLMDataCenterSimulator(all_fullkv_agent)
    # simulator = LLMDataCenterSimulator(all_snapkv_64_agent)
    # simulator = LLMDataCenterSimulator(rule_based_by_price)
    simulator.run_simulator()
