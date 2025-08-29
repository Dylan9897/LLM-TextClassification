import json
from tqdm import tqdm
from sklearn import metrics
import transformers
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from module.adapter import create_and_replace



from transformers import AutoModelForSequenceClassification,AutoTokenizer
from module.argument import ModelArguments,DataArguments,TrainingArguments,LoraArguments

parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

(
    model_args,
    data_args,
    training_args,
    lora_args,
) = parser.parse_args_into_dataclasses()

device_map = None

# Set RoPE scaling factor
config = transformers.AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    trust_remote_code=True,
    is_training=model_args.is_training
)
config.use_cache = False
print(f"checkpoint for config is {config}")

model = AutoModelForSequenceClassification.from_pretrained("output_qwen/longnews",num_labels=7)
tokenizer = AutoTokenizer.from_pretrained("output_qwen/longnews")
model.config.pad_token_id = 151643
model.cuda()

lora_config = LoraConfig(
    r=lora_args.lora_r,
    lora_alpha=lora_args.lora_alpha,
    target_modules=lora_args.lora_target_modules,
    lora_dropout=lora_args.lora_dropout,
    bias=lora_args.lora_bias,
    # task_type="SEQ_CLS"
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
create_and_replace(model)
# print(model)
# s = input()
# 加载保存的权重
weights = load_file("output_qwen/longnews/adapter_model.safetensors")
model.load_state_dict(weights, strict=False)

y_true = []
y_pred = []
with open("data/datasets/longnews/dev.json","r",encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        example = json.loads(line)
        content = example["content"]
        label = eval(example["label"])
        y_true.append(label)
        input_demo = tokenizer(content, padding="max_length",truncation=True,return_tensors="pt")
        for key in input_demo.keys():
            input_demo[key] = input_demo[key].cuda()
        output = model(**input_demo)
        pred = output.logits.argmax().item()
        y_pred.append(pred)

report = metrics.classification_report(y_true, y_pred)
print(report)











