import ollama

response = ollama.list()

res = ollama.chat(
    model="deepseek-r1:1.5b",
    messages=[{"role":"user", "content":"who is iron man"}],
                 stream=True)

# print(res['message']['content'])

for chunk in res:
    print(chunk['message']['content'], end="", flush=True)


###
res = ollama.generate(
    model="deepseek-r1:1.5b",
    prompt="why is the sky blue?"
)

print(res['response'])

####

system = """SYSTEM You are very smart assistant who knows about Marvel Cinematic Universe"""

ollama.create(model='Jarvis', from_='llama2', system=system, parameters={'temperatur':1.0})
print(ollama.list())

res = ollama.generate(model='Jarvis', prompt="Who is iron man?")
print(res['response'])

ollama.delete("Jarvis")