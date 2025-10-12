import requests

def generate_answer(query, context):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt},
        stream=True
    )
    full_output = ""
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            full_output += decoded + "\n"
    return full_output

if __name__ == "__main__":
    query = "Explain the process of photosynthesis."
    context = "Photosynthesis is the process by which plants convert light energy into chemical energy."
    answer = generate_answer(query, context)
    print("Generated Answer:\n", answer)
