import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langgraph_flow import run_flow

def main():
    print("Welcome to CogniGraph CLI!")
    while True:
        user_query = input("\nEnter your query (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        result = run_flow(user_query)
        print(f"[CogniGraph] Action: {result.get('action')}")
        if 'summary' in result:
            print(f"[Summary]: {result['summary']}")
        elif 'documents' in result:
            print(f"[Documents]: {result['documents']}")
        else:
            print("[No result]")

if __name__ == "__main__":
    main()
