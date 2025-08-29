import json
from tqdm import tqdm
from sklearn import metrics
from transformers import AutoModelForSequenceClassification,AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("output_qwen/thucnews")
tokenizer = AutoTokenizer.from_pretrained("output_qwen/thucnews")
model.cuda()




y_true = []
y_pred = []
with open("data/datasets/thucnews/test.json","r",encoding="utf-8") as f:
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

columns = open("data/datasets/thucnews/class.txt","r",encoding="utf-8").readlines()
columns = [x.strip("\n") for x in columns]

report = metrics.classification_report(y_true, y_pred, target_names=columns)
print(report)
