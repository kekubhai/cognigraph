TEST_QUERIES = [
    {
        "query": "Who founded the Mauryan Empire?",
        "expected_answer": "Chandragupta Maurya",
        "expected_topics": ["Chandragupta", "Mauryan Empire"],
    },
    {
        "query": "What technology stack does CogniGraph use?",
        "expected_answer": "LangGraph, LangChain, ChromaDB, Hugging Face Transformers, and Sentence Transformers",
        "expected_topics": ["LangGraph", "LangChain", "ChromaDB", "Hugging Face", "Sentence Transformers"],
    },
    {
        "query": "How do I install CogniGraph dependencies?",
        "expected_answer": "pip install -r requirements.txt",
        "expected_topics": ["pip", "requirements.txt"],
    },
    {
        "query": "Who was Chanakya?",
        "expected_answer": "Chanakya was an orthodox Brahmin teacher from Taxila who met Chandragupta Maurya. Also known as Kautilya and Vishnugupta.",
        "expected_topics": ["Chanakya", "Kautilya", "Taxila", "Brahmin"],
    },
    {
        "query": "What are the deployment options for CogniGraph?",
        "expected_answer": "Local development with Python virtual environments, containerized deployment with Docker and Docker Compose, CLI interface, YAML configuration, environment-based configuration.",
        "expected_topics": ["Docker", "Docker Compose", "CLI", "YAML", "Python"],
    },
    {
        "query": "What is the role of the Planner Agent?",
        "expected_answer": "The Planner Agent is an intelligent decision-making component that analyzes user queries and determines what action to take.",
        "expected_topics": ["Planner Agent", "decision-making", "analyzes"],
    },
    {
        "query": "What embedding model does CogniGraph use by default?",
        "expected_answer": "sentence-transformers/all-MiniLM-L6-v2",
        "expected_topics": ["all-MiniLM-L6-v2", "sentence-transformers"],
    },
    {
        "query": "What was the Nanda Dynasty?",
        "expected_answer": "The Nanda Dynasty was a formidable power in northern India that ruled from Pataliputra before Chandragupta Maurya's rise to power.",
        "expected_topics": ["Nanda", "Pataliputra", "northern India"],
    },
    {
        "query": "What are the use cases of CogniGraph?",
        "expected_answer": "Research assistance, knowledge base querying, information extraction, automated summarization, and interactive Q&A systems.",
        "expected_topics": ["research", "knowledge base", "summarization", "Q&A"],
    },
    {
        "query": "Where did Chandragupta spend his youth?",
        "expected_answer": "He spent his formative youth in Taxila.",
        "expected_topics": ["Taxila", "youth"],
    },
]
