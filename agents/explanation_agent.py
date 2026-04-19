"""Tutor agent: structured explanations from retrieved context."""

from langchain_core.prompts import ChatPromptTemplate

from agents.groq_llms import chat_groq_70b

EXPLAIN_PROMPT = ChatPromptTemplate.from_template("""
You are a brilliant, patient tutor. Using the context below, explain the topic clearly.
If the context says "Use general knowledge", rely on your own expertise.

Structure your response:
1. Brief definition
2. How it works (step by step if needed)
3. Real-world analogy or example

Topic: {topic}
Context: {context}

Explanation:
""")


def explain(topic: str, context: str) -> str:
    chain = EXPLAIN_PROMPT | chat_groq_70b()
    return chain.invoke({"topic": topic, "context": context}).content
