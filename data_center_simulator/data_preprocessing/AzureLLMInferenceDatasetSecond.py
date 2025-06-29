import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def get_invocations_per_second(df):
    df_copy = df.copy()
    df_copy["Time"] = df_copy["TIMESTAMP"].dt.round(freq="s")
    df_res = df_copy.groupby("Time").count()["TIMESTAMP"]
    return df_res


if __name__ == "__main__":
    TRACE_NAMES = [
        "Coding",
        "Conversation",
    ]

    # 2023
    TRACE_FILENAMES = [
        "../data/LLM_inference_trace/raw/2023/AzureLLMInferenceTrace_code.csv",
        "../data/LLM_inference_trace/raw/2023/AzureLLMInferenceTrace_conv.csv",
    ]
    # Read all traces
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"])
        df_res = get_invocations_per_second(df_traces[trace_name])
        # save to csv
        filepath = f"../data/LLM_inference_trace/processed/2023/invocations_per_second_{trace_name}.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_res.to_csv(filepath, index=True)
        # save plot
        df_res.plot(grid=True, ylim=0, figsize=(15, 5))
        plt.ylabel("Number of invocations per minute")
        plt.tight_layout()
        plt.savefig(f"../data/LLM_inference_trace/processed/2023/invocations_per_second_{trace_name}.png")
        plt.close()

    # 2024
    TRACE_FILENAMES = [
        "../data/LLM_inference_trace/raw/2024/AzureLLMInferenceTrace_code_1week.csv",
        "../data/LLM_inference_trace/raw/2024/AzureLLMInferenceTrace_conv_1week.csv",
    ]
    # Read all traces
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"], date_format="mixed", dayfirst=False)
        df_res = get_invocations_per_second(df_traces[trace_name])
        # save to csv
        filepath = f"../data/LLM_inference_trace/processed/2024/invocations_per_second_{trace_name}.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_res.to_csv(filepath, index=True, date_format='%Y-%m-%d %H:%M:%S')
        # save plot
        df_res.plot(grid=True, ylim=0, figsize=(15, 5))
        plt.ylabel("Number of invocations per minute")
        plt.tight_layout()
        plt.savefig(f"../data/LLM_inference_trace/processed/2024/invocations_per_second_{trace_name}.png")
        plt.close()
