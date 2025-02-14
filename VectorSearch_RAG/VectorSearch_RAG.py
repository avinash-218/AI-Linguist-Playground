import pymongo
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MONGODB_CLIENT = 'mongodb+srv://username:password@genai.gkfob.mongodb.net/?retryWrites=true&w=majority&appName=GenAI'

client = pymongo.MongoClient(MONGODB_CLIENT)
db = client.sample_mflix
collection = db.movies

# Generate and Push Embeddings to MongoDB
def generate_embedding(payload, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(payload)
    return embeddings.tolist()

# for doc in tqdm(collection.find({'plot': {'$exists': True}}).limit(50)):
#     doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#     collection.replace_one({'_id': doc['_id']}, doc)

# Vector Search
query = "imaginary characters from outer space at war"

results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "plot_embedding_hf",
        "numCandidates": 100,
        "limit": 4,
        "index": "PlotSemanticSearch"
    }}
])

for doc in results:
    print(f'Movie Name: {doc["title"]}, \nMovie Plot: {doc["plot"]}\n')