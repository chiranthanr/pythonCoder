from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="mistral")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Python programmer who is proficient in all aspects of the python language. You will "
                "act as an assistant to help with writing code based on any use case given to you."
            ),
            ("human","{question}")
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable",runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):
        await msg.stream_token(chunk)
    await msg.send()

