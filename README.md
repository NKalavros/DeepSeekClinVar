# DeepSeek ClinVar

Using a distilled DeepSeek, ollama and DSPy to create an automated task of structured knowledge extraction against existing knowledgebases.

Given a database of articles and relationships, can it answer questions about the relationship of two proteins/a phenotype and a variant, etc.

Get to working by Jan 3.

## Installation

1. Install [ollama](https://github.com/ollama/ollama/) (Using 0.5.7 here)

```
wget https://github.com/ollama/ollama/releases/download/v0.5.7/ollama-linux-amd64.tgz
```
```
tar -zxvf ollama-linux-amd64.tgz
```

2. Install DSPy

```
pip install dspy
```

3. Deploy a model (here DeepSeek r1 14b distilled based on Llama and Qwen)

I'm only using a single node here, so we'll have to burden it. Otherwise consult the ollama docs on how to serve this.

```
ollama run deepseek-r1:14b
```



