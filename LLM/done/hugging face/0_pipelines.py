from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))
print(classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]))

# Audio Classification
classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
print(classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"))