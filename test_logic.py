
import logic
import time

def test_search():
    print("Initializing retriever...")
    logic.initialize_retriever()
    
    # Mock data
    animals = [
        {"name": "Lion", "description": "A large wild cat known as the king of the jungle.", "diet": "Carnivore", "habitat": "Savanna"},
        {"name": "Penguin", "description": "A flightless bird that lives in the cold.", "diet": "Carnivore", "habitat": "Antarctica"},
        {"name": "Elephant", "description": "A huge mammal with a trunk.", "diet": "Herbivore", "habitat": "Savanna"}
    ]
    
    print("Setting data...")
    logic.set_animals_data(animals)
    
    query = "big cat"
    print(f"Searching for: '{query}'")
    results = logic.search_animals(query)
    
    print("Results:")
    for r in results:
        print(f"- {r['name']} (Score: {r.get('_score', 0):.4f})")

if __name__ == "__main__":
    test_search()
