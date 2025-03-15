from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    max_new_tokens=512,
    do_sample=False)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("Who are you?")
print(result.content)
