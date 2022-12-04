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
