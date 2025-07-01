from transformers import pipeline

class SummarizerAgent:
    def __init__(self, model_name: str, hf_token: str = None):
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            token=hf_token,
            truncation=True
        )

    def summarize(self, chunks, user_query):
        context = "\n".join(chunks)
        prompt = f"Summarize the following in the context of the user query: {user_query}\n{context}"
        result = self.summarizer(prompt, max_length=256, min_length=32, do_sample=False)
        return result[0]["summary_text"] if result and "summary_text" in result[0] else ""
