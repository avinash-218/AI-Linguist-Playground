import os
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain

# Neo4j credentials and connection details
NEO4J_URI = "neo4j+s://b315cc04.databass.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "-Ibwlw_k4"

# Groq API key for LLM usage
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxx"

# Set environment variables so LangChain's Neo4j connector can pick them up
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

def init_graph():
    """Initialize and return a Neo4jGraph connection object."""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )

def init_llm():
    """Initialize and return a ChatGroq LLM instance."""
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

def create_graph_documents(text, llm):
    """
    Convert a block of text into a graph-structured format
    using the LLMGraphTransformer.

    Args:
        text (str): Input text to be transformed.
        llm: LLM instance (ChatGroq in this case).

    Returns:
        list: GraphDocument objects ready for insertion into Neo4j.
    """
    documents = [Document(page_content=text)]
    llm_transformer = LLMGraphTransformer(llm=llm)
    return llm_transformer.convert_to_graph_documents(documents)

def load_movie_dataset(graph):
    """
    Load a CSV dataset of movies into Neo4j and create nodes + relationships.

    Args:
        graph: An active Neo4jGraph connection.
    """
    movie_query = """
    LOAD CSV WITH HEADERS FROM
    'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv' as row

    MERGE(m:Movie{id:row.movieId})
    SET m.released = date(row.released),
        m.title = row.title,
        m.imdbRating = toFloat(row.imdbRating)

    // Link directors
    FOREACH (director in split(row.director, '|') |
        MERGE (p:Person {name:trim(director)})
        MERGE (p)-[:DIRECTED]->(m))

    // Link actors
    FOREACH (actor in split(row.actors, '|') |
        MERGE (p:Person {name:trim(actor)})
        MERGE (p)-[:ACTED_IN]->(m))

    // Link genres
    FOREACH (genre in split(row.genres, '|') |
        MERGE (g:Genre {name:trim(genre)})
        MERGE (m)-[:IN_GENRE]->(g))
    """
    # Run the Cypher load script
    graph.query(movie_query)

    # Refresh schema after adding data
    graph.refresh_schema()
    print(graph.schema)

def run_sample_queries(chain):
    """
    Run natural language queries via GraphCypherQAChain
    and print results.

    Args:
        chain: An initialized GraphCypherQAChain object.
    """
    queries = [
        "Who was the director of the movie GoldenEye",
        "Tell me the genre of the movie GoldenEye",
        "Who was the director in movie Casino",
        "Which movies were released in 2008",
        "Give me the list of movies having imdb rating more than 8"
    ]
    for q in queries:
        print(f"Q: {q}")
        print(chain.invoke({"query": q}))

def run_example_cypher_queries(graph):
    """
    Run pre-defined Cypher queries directly on Neo4j
    for testing and debugging purposes.

    Args:
        graph: An active Neo4jGraph connection.
    """
    examples = [
        {"question": "How many artists are there?",
         "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)"},
        {"question": "Which actors played in the movie Casino?",
         "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name"},
        {"question": "How many movies has Tom Hanks acted in?",
         "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)"},
        {"question": "List all the genres of the movie Schindler's List",
         "query": "MATCH (m:Movie {title: 'Schindler\\'s List'})-[:IN_GENRE]->(g:Genre) RETURN g.name"},
        {"question": "Which actors have worked in movies from both the comedy and action genres?",
         "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name"},
        {"question": "Which directors have made movies with at least three different actors named 'John'?",
         "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name"},
        {"question": "Identify movies where directors also played a role in the film.",
         "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name"},
        {"question": "Find the actor with the highest number of movies in the database.",
         "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1"},
    ]

    for ex in examples:
        print(f"Q: {ex['question']}")
        print(graph.query(ex['query']))

if __name__ == "__main__":
    # Step 1: Connect to Neo4j and initialize LLM
    graph = init_graph()
    llm = init_llm()

    # Step 2: Example free-text to graph conversion
    text = """
    Elon Reeve Musk (born June 28, 1971) is a businessman and investor known for his key roles in space
    company SpaceX and automotive company Tesla, Inc. Other involvements include ownership of X Corp.,
    formerly Twitter, and his role in the founding of The Boring Company, xAI, Neuralink and OpenAI.
    He is one of the wealthiest people in the world; as of July 2024, Forbes estimates his net worth to be
    US$221 billion.Musk was born in Pretoria to Maye and engineer Errol Musk, and briefly attended
    the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through
    his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada.
    Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics
    and physics. He moved to California in 1995 to attend Stanford University, but dropped out after
    two days and, with his brother Kimbal, co-founded online city guide software company Zip2.
    """
    graph_docs = create_graph_documents(text, llm)
    graph.add_graph_documents(graph_docs)

    # Step 3: Load movies dataset from CSV into Neo4j
    load_movie_dataset(graph)

    # Step 4: Create the QA chain for Cypher generation from natural language
    chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

    # Step 5: Run both LLM-driven and manual Cypher queries
    run_sample_queries(chain)
    run_example_cypher_queries(graph)
