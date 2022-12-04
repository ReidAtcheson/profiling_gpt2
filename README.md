# profiling_gpt2
Some tools and workflow I used to profile gpt2 model from huggingface checkpoint


# Cloud instance and image used

I used a prebaked nvidia image on GCP with an nvidia A100 - screengrab below
![image](https://user-images.githubusercontent.com/2857424/205463692-38331e38-8865-4db6-97c8-4c5fadaa1f78.png)


# Model & checkpoint used and conversion to onnx

I used huggingface GPT2 checkpoints for `gpt2-huge` and converted this to onnx with the torch onnx module.

The rough script for this was

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, onnx
from torchviz import make_dot
import torch
import torch.nn as nn
import onnx
# Configuration
# Load codeparrot tokenizer trained for Python code tokenization
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
# Config: "scale_attn_by_layer_idx" and "reorder_and_upcast_attn" are Mistral stability tweaks
config_kwargs = {
    "vocab_size": len(tokenizer),
    "scale_attn_by_inverse_layer_idx": True,
    "reorder_and_upcast_attn": True,
}
# Load model config (GPT-2 large in this case)
config = AutoConfig.from_pretrained("gpt2-large", **config_kwargs)
# Initialize new model with config
model = AutoModelForCausalLM.from_config(config)
#Make a fake module to turn off key,value cache so these do not
#get traced during onnx export
class TmpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmp=model
    def forward(self,x):
        return self.tmp(x,use_cache=False)
tmpmodel=TmpModel()
torch.onnx.export(tmpmodel,               # model being run
                  torch.randint(1, len(tokenizer),(1,1024)),  # example input for the model
                  "simple_model.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=15,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'], # the model's output names
                  training=torch.onnx.TrainingMode.EVAL
                  )
```



# Converting ONNX to TRT Engine

To get fp32 and fp16 engines with extra profiling information like NVTX markers (for use in `nsys`) I did the following:

```
trtexec --onnx=simple_model.onnx --saveEngine=gpt2_fp32.engine --profilingVerbosity=detailed --workspace=30000
trtexec --onnx=simple_model.onnx --fp16 --saveEngine=gpt2_fp16.engine --profilingVerbosity=detailed --workspace=30000
```



# Basic timings with trtexec

`trtexec --loadEngine=gpt2_fp32.engine --useCudaGraph`

```
[12/04/2022-02:01:36] [I] === Trace details ===
[12/04/2022-02:01:36] [I] Trace averages of 10 runs:
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0561 ms - Host latency: 41.2429 ms (enqueue 0.151791 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0865 ms - Host latency: 41.274 ms (enqueue 0.148151 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0933 ms - Host latency: 41.2813 ms (enqueue 0.149274 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0886 ms - Host latency: 41.2748 ms (enqueue 0.137341 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0964 ms - Host latency: 41.2835 ms (enqueue 0.13833 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0831 ms - Host latency: 41.2698 ms (enqueue 0.155908 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.086 ms - Host latency: 41.2733 ms (enqueue 0.140601 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0899 ms - Host latency: 41.2763 ms (enqueue 0.148071 ms)
[12/04/2022-02:01:36] [I] Average on 10 runs - GPU latency: 31.0868 ms - Host latency: 41.2747 ms (enqueue 0.151758 ms)
[12/04/2022-02:01:36] [I] 
[12/04/2022-02:01:36] [I] === Performance summary ===
[12/04/2022-02:01:36] [I] Throughput: 31.8446 qps
[12/04/2022-02:01:36] [I] Latency: min = 41.1393 ms, max = 41.3042 ms, mean = 41.273 ms, median = 41.2759 ms, percentile(90%) = 41.2895 ms, percentile(95%) = 41.2952 ms, percentile(99%) = 41.3042 ms
[12/04/2022-02:01:36] [I] Enqueue Time: min = 0.120361 ms, max = 0.22583 ms, mean = 0.148609 ms, median = 0.146973 ms, percentile(90%) = 0.173096 ms, percentile(95%) = 0.180908 ms, percentile(99%) = 0.22583 ms
[12/04/2022-02:01:36] [I] H2D Latency: min = 0.00842285 ms, max = 0.0283203 ms, mean = 0.0102436 ms, median = 0.00927734 ms, percentile(90%) = 0.012207 ms, percentile(95%) = 0.0171509 ms, percentile(99%) = 0.0283203 ms
[12/04/2022-02:01:36] [I] GPU Compute Time: min = 30.9525 ms, max = 31.1133 ms, mean = 31.0858 ms, median = 31.0886 ms, percentile(90%) = 31.104 ms, percentile(95%) = 31.1082 ms, percentile(99%) = 31.1133 ms
[12/04/2022-02:01:36] [I] D2H Latency: min = 10.1726 ms, max = 10.1826 ms, mean = 10.1769 ms, median = 10.1768 ms, percentile(90%) = 10.1783 ms, percentile(95%) = 10.1787 ms, percentile(99%) = 10.1826 ms
[12/04/2022-02:01:36] [I] Total Host Walltime: 3.10885 s
[12/04/2022-02:01:36] [I] Total GPU Compute Time: 3.0775 s
[12/04/2022-02:01:36] [I] Explanations of the performance metrics are printed in the verbose logs.
```

# Getting nsight systems trace

For the trace I did the following 
```
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx trtexec --iterations=1 --loadEngine=gpt2_fp32.engine
```

# Getting performance counters with nsight compute

To get performance counters I did

```
sudo env "PATH=$PATH" ncu --section ComputeWorkloadAnalysis --csv trtexec --loadEngine=gpt2_fp32.engine > cwa.csv
sudo env "PATH=$PATH" ncu --section SpeedOfLight --csv trtexec --loadEngine=gpt2_fp32.engine > sol.csv
sudo env "PATH=$PATH" ncu --section Occupancy --csv trtexec --loadENgine=gpt2_fp32.engine > occupancy.csv
```

# Extracting kernels and start,stop times from nsys systems trace sqlite file

```
sqlite3 -csv report2.sqlite 'SELECT names.value AS name, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' > kernels.csv
```

# Combining perf counters and kernels into a single trace file


```python
import pandas as pd
df=pd.read_csv("kernels.csv")
counters=pd.read_csv("sol.csv")
occupancy=pd.read_csv("occupancy.csv")
d={}
for name,metric,value in zip(counters["Kernel Name"],counters["Metric Name"],counters["Metric Value"]):
    if name in d:
        if metric in d[name]:
            d[name][metric]=value
        else:
            d[name][metric]=value
    else:
        d[name] = {metric : value}
for name,metric,value in zip(occupancy["Kernel Name"],occupancy["Metric Name"],occupancy["Metric Value"]):
    if name in d:
        if metric in d[name]:
            d[name][metric]=value
        else:
            d[name][metric]=value
f=open("trace.json","w")
f.write("[")
last=len(df["name"])
for i,(name,start,stop) in enumerate(zip(df["name"],df["start"],df["stop"])):
    start=start/1000
    stop=stop/1000
    json=f"""{{
    "name": "{name}", 
    "cat": "foo", 
    "ph": "B",
    "ts": {start},
    "pid": 1,
    "tid": 1,
         "args": {{
        "first": 1
     }}
    }},
    {{
    "ph": "E", 
    "ts": {stop},
    "pid": 1, 
    "tid": 1,
     "args": {{
       "first": 4,
       "second": 2
     }}
    }},
    """
    counter1=f"""{{
    "pid" : "Memory [%]",
    "name": "Memory [%]", 
    "ph": "C", 
    "ts":  {start}, 
    "args": {{
        "Memory [%]":  {d[name]["Memory [%]"]}
        }}
    }},
    """
    counter2=f"""{{
    "pid" : "Compute (SM) [%]",
    "name": "Compute (SM) [%]",
    "ph": "C", 
    "ts":  {start}, 
    "args": {{
        "Memory [%]":  {d[name]["Compute (SM) [%]"]}
        }}
    }},
    """
   counter3=f"""{{
    "pid" : "Theoretical Occupancy",
    "name": "Theoretical Occupancy",
    "ph": "C", 
    "ts":  {start}, 
    "args": {{
        "Memory [%]":  {d[name]["Theoretical Occupancy"]}
        }}
    }},
    """
    counter4=f"""{{
    "pid" : "Achieved Occupancy",
    "name": "Achieved Occupancy",
    "ph": "C", 
    "ts":  {start}, 
    "args": {{
        "Memory [%]":  {d[name]["Achieved Occupancy"]}
        }}
    }}{"" if i==last-1 else ","}
    """




    f.write(json)
    f.write(counter1)
    f.write(counter2)
    f.write(counter3)
    f.write(counter4)

                                                     
```

The resulting file can be opened with `chrome://tracing` and looks something like below:

![image](https://user-images.githubusercontent.com/2857424/205469424-401f45a2-3d28-4a73-ba7d-ae09fabb5d4c.png)

# Converting ONNX to Intel inference engine via Openvino

```
mo --input_model simple_model.onnx 
```

makes the engine

# Benchmarking OpenVino engine
```
benchmark_app -m simple_model.xml -hint latency -ip f32 -op f32 -report_type detailed_counters
```


```
Latency:
    Median:     2101.50 ms
    AVG:        2100.02 ms
    MIN:        2085.78 ms
    MAX:        2126.31 ms

```





# References
 * https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
 * https://docs.nvidia.com/nsight-systems/2020.3/profiling/index.html#cli-profiling
 * https://docs.nvidia.com/nsight-visual-studio-edition/nvtx/index.html
 * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
 * https://huggingface.co/transformers/v4.4.2/model_doc/gpt2.html#gpt2lmheadmodel
 * https://huggingface.co/transformers/v4.4.2/glossary.html#position-ids
 * https://huggingface.co/transformers/v4.4.2/pretrained_models.html
 * https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#
 * https://stackoverflow.com/questions/66626185/how-do-i-get-my-kernel-execution-times-from-the-sqlite3-output-of-nsight-systems
 * https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
 * https://www.intel.com/content/www/us/en/develop/documentation/ei4amr-2022-3-developer-guide/top/tutorials-amr/benchmark_profiling/run-openvino-benchmarking-tool.html
