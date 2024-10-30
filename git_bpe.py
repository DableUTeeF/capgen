from transformers import AutoTokenizer, AutoProcessor, GPT2Tokenizer


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained('microsoft/git-base-coco')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    t2 = GPT2Tokenizer()
    print()
