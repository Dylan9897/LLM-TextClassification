from transformers import AutoModelForSequenceClassification
import sys
sys.path.append('/home/root123/workspace/handx/LLM-TextClassification')
from module.TemporalAttention import TemporalAttention

def create_and_replace(model):
    if hasattr(model, 'score'):
        target_model = model.score
        in_features = target_model.in_features
        out_features = target_model.out_features

        new_model = TemporalAttention(input_seq_len=in_features,output_seq_len=out_features,feature_dim=512)

        setattr(model,"score",new_model)
    else:
        raise Exception("Please confirm whether the name of the layer in the model is correct")



if __name__=="__main__":
    model_path = "ckpt/DeepSeek-R1-Distill-Qwen-1___5B"
    num_labels = 7
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=num_labels)
    create_and_replace(model)
    print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter Name: {name}, Updateable: True")
    #     else:
    #         print(f"Parameter Name: {name}, Updateable: False")