import requests
import sys

def main():
    print("=====================================================")
    print(" Welcome to the Local RAG Knowledge Base CLI - Sumit_AI")
    print("=====================================================")
    print("Ensure that the backend is running in another terminal:")
    print("  python generate_response.py")
    print("\nType 'quit' or 'exit' to stop.")
    print("-----------------------------------------------------")
    
    while True:
        try:
            query = input("\n[You]: ")
            if query.lower() in ('quit', 'exit'):
                break
            
            if not query.strip():
                continue
                
            print("[Sumit_AI]: Thinking...")
            response = requests.post("http://localhost:9000/chat", json={"message": query})
            
            if response.status_code == 200:
                print(f"[Sumit_AI]: {response.json().get('response', '')}")
            else:
                print(f"[System]: Error - {response.status_code} {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("[System Error]: Cannot connect to the server. Is generate_response.py running on port 9000?")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
