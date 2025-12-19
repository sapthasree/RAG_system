def generate_answer(context, query):
    try:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
You are an AI assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}
"""
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        # Fallback when API key / quota / internet fails
        return f"""[FALLBACK ANSWER]

Question:
{query}

Based on retrieved context:
{context[:500]}
"""
