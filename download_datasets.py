# --- Shakespeare 데이터셋 다운로드 ---
import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/data/shakespeare.txt"

response = requests.get(url)

with open("data/shakespeare.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

print("shakespeare.txt 생성 완료")



# --- TinyStories 데이터셋 다운로드 ---
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

stories = []

for i in range(5000):
    stories.append(dataset["train"][i]["text"])

text = "\n\n".join(stories)

with open("data/tinystories.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("tinystories.txt 생성 완료")