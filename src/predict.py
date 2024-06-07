from transformers import TFAutoModelForSeq2SeqLM

from src.utils import tokenizer


def predict(sentence: str, checkpoints_path: str):
    inputs = tokenizer(sentence, return_tensors="tf")["input_ids"]
    print(inputs)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoints_path)
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
