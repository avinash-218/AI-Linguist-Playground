import phi.api
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key=os.getenv('OPENAI_API_KEY')

phi.api = os.getenv('PHI_API_KEY')

web_search_agent  = Agent(
    name = "Jarvis",
    role = 'Search the web for the information',
    model = Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tools_calls = True,
    markdown = True
)

financial_agent = Agent(
    name = "Harshad Mehta",
    model = Groq(id="llama-3.3-70b-versatile"),
    tool = [
        YFinanceTools(
            stock_price = True,
            analyst_recommendations = True,
            stock_fundamentals = True,
            company_news = True)],
    instructions = ["Use tables to display the data"],
    show_tools_calls = True,
    markdown = True
)

app = Playground(agents=[web_search_agent, financial_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app", reload=True)