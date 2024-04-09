import os
import fitz
import glob
from tqdm import tqdm

from transformers import AutoTokenizer
from huggingface_hub import InferenceClient

client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=os.environ["HF_TOKEN"])
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

def get_text_first_page(pdf_path):
    pdf_document = fitz.open(pdf_path)
    first_page = pdf_document[0]
    text = first_page.get_text()
    return text


def create_summary(pdf_file):
    if os.path.exists(os.path.dirname(pdf_file) + '/summary.txt'):
        return
    text = get_text_first_page(pdf_file)
    chat = [{"role": "user", "content": "Summary of the paper in 3-6 bullet sentences.\n\n" + text}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    summary = client.text_generation(prompt, max_new_tokens=512)
    open(os.path.dirname(pdf_file) + '/summary.txt', 'w').write(summary)


def main():
    pdf_files = glob.glob('papers/**/*.pdf')

    for pdf_file in tqdm(pdf_files):
        create_summary(pdf_file)

if __name__ == "__main__":
    main()