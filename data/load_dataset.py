from datasets import load_dataset

# Loading the rag-mini-wikipedia dataset
dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

# Printing the info to verify it loaded
print(dataset)
print("Example entry:", dataset["test"][0])
