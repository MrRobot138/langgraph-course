from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. "
            "Generate critique and recommendations for the user's tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages([
    ( "system",
      "You are a viral twitter influencer generating a tweet based on user specifications. "
      "Generate a tweet that is engaging, concise, and likely to go viral. "
      "Incorporate any specific themes, topics or general recommendatios requested by the user."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

llm = ChatOpenAI(model_name="gpt-5-mini")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
