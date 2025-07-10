from interface.cli import main as cli_main
from data.ingest import ingest_documents
import os

def main():
    """Main entry point that ingests documents and then starts the CLI."""
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Ingest documents first
    print("Ingesting documents...")
    ingest_documents(data_dir)
    
    # Start the CLI
    print("Starting CogniGraph CLI...")
    cli_main()

if __name__ == "__main__":
    main()
