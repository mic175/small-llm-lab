import csv
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

from transformers import pipeline


# 1. Define the experiment settings in one place.
# 集中设置实验参数。
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
max_new_tokens = 32
num_runs = 5
prompt = "Explain what a GPU is in two short sentences."


# 2. Load the model and tokenizer once, using the CPU.
# 加载一次模型和 tokenizer，使用 CPU。
print("Loading model...")

generator = pipeline(
    task="text-generation",
    model=model_name,
    device=-1,
)

messages = [
    {
        "role": "user",
        "content": prompt,
    }
]


# 3. Warm up once without recording the result.
# 预热一次，不计入正式测量。
print("Warming up...")

generator(
    messages,
    max_new_tokens=max_new_tokens,
    do_sample=False,
)


# 4. Measure repeated requests.
# 重复测量，并保存耗时和完整记录。
response_times = []
records = []

for run_number in range(1, num_runs + 1):
    start_time = perf_counter()

    result = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    end_time = perf_counter()
    elapsed_seconds = end_time - start_time

    # Extract and record the result outside the timed interval.
    # 在计时结束后提取并记录结果。
    answer = result[0]["generated_text"][-1]["content"]
    response_times.append(elapsed_seconds)

    records.append(
        {
            "run_number": run_number,
            "model": model_name,
            "device": "cpu",
            "max_new_tokens": max_new_tokens,
            "prompt": prompt,
            "response_seconds": elapsed_seconds,
            "answer": answer,
        }
    )

    print(f"Run {run_number}: {elapsed_seconds:.3f} seconds")


# 5. Display the final response and timing summary.
# 显示最后一次回答及耗时统计。
print("\nLast response:")
print(answer)

print("\nTiming summary:")
print(f"Median: {median(response_times):.3f} seconds")
print(f"Minimum: {min(response_times):.3f} seconds")
print(f"Maximum: {max(response_times):.3f} seconds")


# 6. Create a results folder beside this script.
# 在脚本所在目录下创建 results 文件夹。
project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

# Include date, time, and microseconds in the filename.
# 文件名包含日期、时间和微秒，区分不同次实验。
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
csv_path = results_folder / f"cpu_benchmark_{timestamp}.csv"


# 7. Save one row per measured request.
# 每次正式测量保存为一行。
fieldnames = [
    "run_number",
    "model",
    "device",
    "max_new_tokens",
    "prompt",
    "response_seconds",
    "answer",
]

# newline="" lets the csv module handle line endings correctly.
# utf-8-sig supports Chinese text and convenient opening in Excel.
# newline="" 让 csv 模块处理换行；utf-8-sig 方便 Excel 识别中文。
with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved {len(records)} measurements to:")
print(csv_path)