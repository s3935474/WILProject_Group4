from evaluate import load

# example predicted and reference answers
preds = ["Photosynthesis converts light energy into chemical energy."]
refs = ["Photosynthesis is the process by which plants use sunlight to make food."]

# loading the metrics
bertscore = load("bertscore")
rouge = load("rouge")

# calculating the metrics
bert_result = bertscore.compute(predictions=preds, references=refs, lang="en")
rouge_result = rouge.compute(predictions=preds, references=refs)

print("BERTScore:", bert_result)
print("ROUGE:", rouge_result)
