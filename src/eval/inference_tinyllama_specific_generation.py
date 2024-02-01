from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed_text", type=str)
    return parser.parse_args()


def main(args):
    model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
    model = AutoModelForCausalLM.from_pretrained("models/tinyllama/merged")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250
    )

    result = pipe(args.seed_text)
    output = result[0]["generated_text"]

    print(output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
