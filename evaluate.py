import os
import json
import torch
from transformers import (
    BloomForCausalLM,
    GPT2LMHeadModel,
    XGLMForCausalLM,
    BertForMaskedLM,
    XLMRobertaForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer
)
from minicons import scorer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(scoring_function, suite_name, suite):
    correct = 0
    for item in suite:
        logprobs = scoring_function(item[0], item[1])
        difference = logprobs[0] - logprobs[1]
        if difference > 0:
            correct += 1
    print(f"{suite_name}: {correct / len(suite)}")


def evaluate(model, tokenizer, suite_name, suite):
    if isinstance(model, (BloomForCausalLM, GPT2LMHeadModel, XGLMForCausalLM)):
        scr = scorer.IncrementalLMScorer(model=model, device="auto", tokenizer=tokenizer)
        accuracy(lambda x, y: scr.partial_score(x, y), suite_name, suite)
    elif isinstance(model, (BertForMaskedLM, XLMRobertaForMaskedLM)):
        scr = scorer.MaskedLMScorer(model=model, device="auto", tokenizer=tokenizer)
        accuracy(lambda x, y: scr.partial_score(x, y, PLL_metric="within_word_l2r"), suite_name, suite)
    else:
        raise TypeError("Model type not supported")


if __name__ == "__main__":
    model_name = "bigscience/bloom"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    for suite_name in os.listdir("suites"):
        suite = json.load(open(suite_name))
        evaluate(model, tokenizer, suite_name, suite)