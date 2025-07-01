import sys
from langgraph_flow import run_flow

def main():
    print("Welcome to CogniGraph CLI!")
    while True:
        user_query = input("\nEnter your query (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        summary = run_flow(user_query)
        print(f"\n[CogniGraph] Summary:\n{summary}")
        feedback = input("\nYour feedback (optional, press Enter to skip): ")
        # Feedback could be logged to memory if needed
        print("---")

if __name__ == "__main__":
    main()
