import os
import pandas as pd
import matplotlib.pyplot as plt


def get_invocations_per_second(df, freq='1s'):
    df_copy = df.copy()
    df_copy["Time"] = df_copy["TIMESTAMP"].dt.round(freq=freq)
    df_final = df_copy.groupby("Time").count()["TIMESTAMP"]
    return df_final


if __name__ == "__main__":
    TRACE_NAMES = [
        "Coding",
        "Conversation",
    ]

    # 2023
    TRACE_FILENAMES = [
        "../../data/LLM_inference_trace/raw/2023/AzureLLMInferenceTrace_code.csv",
        "../../data/LLM_inference_trace/raw/2023/AzureLLMInferenceTrace_conv.csv",
    ]
    # Read all traces (1s)
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"])
        df_res = get_invocations_per_second(df_traces[trace_name], freq='1s')
        # save to csv
        filepath = f"../../data/llm_inference_trace/processed/2023/invocations_per_second_{trace_name}.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_res.to_csv(filepath, index=True)
        # save plot
        df_res.plot(grid=True, ylim=0, figsize=(15, 5))
        plt.ylabel("Number of invocations 1 second")
        plt.tight_layout()
        plt.savefig(f"../../data/LLM_inference_trace/processed/2023/invocations_per_second_{trace_name}.png")
        plt.close()
    # Read all traces (10s)
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"])
        df_res = get_invocations_per_second(df_traces[trace_name], freq='10s')
        # save to csv
        filepath = f"../../data/llm_inference_trace/processed/2023/invocations_ten_second_{trace_name}.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_res.to_csv(filepath, index=True)
        # save plot
        df_res.plot(grid=True, ylim=0, figsize=(15, 5))
        plt.ylabel("Number of invocations 10 second")
        plt.tight_layout()
        plt.savefig(f"../../data/LLM_inference_trace/processed/2023/invocations_ten_second_{trace_name}.png")
        plt.close()

    # 2024
    TRACE_FILENAMES = [
        "../../data/LLM_inference_trace/raw/2024/AzureLLMInferenceTrace_code_1week.csv",
        "../../data/LLM_inference_trace/raw/2024/AzureLLMInferenceTrace_conv_1week.csv",
    ]
    # Read all traces (1s)
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"], date_format="mixed", dayfirst=False)
        df_res = get_invocations_per_second(df_traces[trace_name], freq='1s')
        # save to csv
        filepath = f"../../data/llm_inference_trace/processed/2024/invocations_per_second_{trace_name}.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_res.to_csv(filepath, index=True, date_format='%Y-%m-%d %H:%M:%S')
        # save plot
        df_res.plot(grid=True, ylim=0, figsize=(15, 5))
        plt.ylabel("Number of invocations 1 second")
        plt.tight_layout()
        plt.savefig(f"../../data/LLM_inference_trace/processed/2024/invocations_per_second_{trace_name}.png")
        plt.close()
    # Read all traces (10s)
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"], date_format="mixed", dayfirst=False)
        df_res = get_invocations_per_second(df_traces[trace_name], freq='10s')
        # save to csv
        filepath = f"../../data/llm_inference_trace/processed/2024/invocations_ten_second_{trace_name}.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_res.to_csv(filepath, index=True, date_format='%Y-%m-%d %H:%M:%S')
        # save plot
        df_res.plot(grid=True, ylim=0, figsize=(15, 5))
        plt.ylabel("Number of invocations 10 second")
        plt.tight_layout()
        plt.savefig(f"../../data/LLM_inference_trace/processed/2024/invocations_ten_second_{trace_name}.png")
        plt.close()
