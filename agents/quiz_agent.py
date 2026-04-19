"""Quiz agent: MCQs from retrieved context."""

from langchain_core.prompts import ChatPromptTemplate

from agents.groq_llms import chat_groq_70b

QUIZ_PROMPT = ChatPromptTemplate.from_template("""
You are an examiner. Based on the context below, generate exactly 3 multiple-choice questions.
If the context says "Use general knowledge", rely on your own expertise.

FORMAT YOUR OUTPUT EXACTLY like this template for every question — no deviations:

---

**Question 1:** <question text here>

- 🅐 <option A>
- 🅑 <option B>
- 🅒 <option C>
- 🅓 <option D>

✅ **Correct Answer:** <A / B / C / D> — <one sentence explanation>

---

**Question 2:** <question text here>

- 🅐 <option A>
- 🅑 <option B>
- 🅒 <option C>
- 🅓 <option D>

✅ **Correct Answer:** <A / B / C / D> — <one sentence explanation>

---

**Question 3:** <question text here>

- 🅐 <option A>
- 🅑 <option B>
- 🅒 <option C>
- 🅓 <option D>

✅ **Correct Answer:** <A / B / C / D> — <one sentence explanation>

---

Topic: {topic}
Context: {context}
""")


def generate_quiz(topic: str, context: str) -> str:
    chain = QUIZ_PROMPT | chat_groq_70b()
    return chain.invoke({"topic": topic, "context": context}).content
