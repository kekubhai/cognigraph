# Memory module for storing queries, feedback, and summaries

class Memory:
    def __init__(self):
        self.history = []

    def log(self, user_query, feedback, summary):
        self.history.append({
            "query": user_query,
            "feedback": feedback,
            "summary": summary
        })

    def get_history(self):
        return self.history
