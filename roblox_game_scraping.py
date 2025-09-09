import requests
import pandas as pd

# Games + universe IDs + genres
games_list = [
    {"id": 4924922222, "genre": "Roleplay / Avatar Sim"},       # Brookhaven RP
    {"id": 2753915549, "genre": "Action / Adventure"},          # Blox Fruits
    {"id": 920587237, "genre": "Roleplay / Collecting"},        # Adopt Me!
    {"id": 1962086868, "genre": "Obby / Parkour"},              # Tower of Hell
    {"id": 142823291, "genre": "Survival / Mystery"},           # Murder Mystery 2
    {"id": 370731277, "genre": "Social Hangout"},               # MeepCity
    {"id": 4623386862, "genre": "Survival Horror"},             # Piggy
    {"id": 10358819857, "genre": "Fighting / Anime"},           # The Strongest Battlegrounds
    {"id": 735030788, "genre": "Roleplay / Fashion"},           # Royale High
    {"id": 6872265039, "genre": "Strategy / Combat"},           # BedWars
    {"id": 6284583030, "genre": "Collecting / Simulation"},     # Pet Simulator X!
    {"id": 185655149, "genre": "Simulation / Roleplay"},        # Welcome to Bloxburg
    {"id": 6516141723, "genre": "Horror / Escape Room"},        # DOORS
    {"id": 286090429, "genre": "FPS / Shooter"},                # Arsenal
    {"id": 606849621, "genre": "Action / Cops & Robbers"}       # Jailbreak
]

# Prepare API call
universe_ids = ",".join(str(g["id"]) for g in games_list)
url = f"https://games.roblox.com/v1/games?universeIds={universe_ids}"

# Fetch game details
response = requests.get(url)
data = response.json().get("data", [])

# Merge genre info
results = []
for g in data:
    match = next((x for x in games_list if x["id"] == g["id"]), {})
    results.append({
        "id": g["id"],
        "name": g.get("name"),
        "active_users": g.get("playing"),
        "visits": g.get("visits"),
        "like_ratio": g.get("likeRatio"),
        "genre": match.get("genre", "Unknown")
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("top_roblox_games.csv", index=False)

print("Data saved to top_roblox_games.csv")
