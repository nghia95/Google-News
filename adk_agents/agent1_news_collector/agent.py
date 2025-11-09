from google.adk.agents import Agent
from google.adk.tools import AgentTool

from .tools import fetch_stock_news_from_google_news, predict_index

google_search_agent = Agent(
    name="google_search_agent",
    model="gemini-2.0-flash",
    instruction="Retrieves recent news articles from Google News.",
    tools=[fetch_stock_news_from_google_news]
)

predict_agent = Agent(
    name="predict_index_price_agent",
    model="gemini-2.0-flash",
    instruction="Predict the closing price of a stock index based on news articles.",
    tools=[predict_index]
)

root_agent = Agent(
    name="news_agent_v1",
    model="gemini-2.0-flash", # Can be a string for Gemini or a LiteLlm object
    description="Analyzes market news to predict stock index trends.",
    instruction="You are a helpful finance assistant. When the user asks for a prediction about a specific index(like DOW or N225), you will provide a prediction about the close price of that index using the following step:   1. Use the 'google_search_agent' tool to find the latest news about that index and output it to the data.json file2. Use the 'predict_agent' tool to provide the prediction of the closed price of the index, pass the name of the index to the 'target_index' parameter 3. Output the prediction in natural language",
    tools=[
        
        AgentTool(google_search_agent, skip_summarization=False),
        AgentTool(predict_agent, skip_summarization=False)
        
    ]
)
