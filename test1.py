import requests
import pandas as pd
import time

rolimons_url = "https://api.rolimons.com/games/v1/gamelist"
rolimons_data = requests.get(rolimons_url).json()
games_dict = rolimons_data.get("games", {})

print(f"Total Rolimons games fetched: {len(games_dict)}")

universe_ids = []
for place_id_str in games_dict.keys():
    place_id = int(place_id_str)
    url = f"https://apis.roblox.com/universes/v1/places/{place_id}/universe"
    resp = requests.get(url).json()
    uni_id = resp.get("universeId")
    if uni_id:
        universe_ids.append(uni_id)
    time.sleep(0.3)  # be kind to Roblox API

print(f"Total Universe IDs mapped: {len(universe_ids)}")

# Now fetch game details in batches
records = []
batch_size = 100
for i in range(0, len(universe_ids), batch_size):
    batch = universe_ids[i:i+batch_size]
    url = "https://games.roblox.com/v1/games?universeIds=" + ",".join(map(str, batch))
    resp = requests.get(url).json()
    data = resp.get("data", [])
    print(f"Batch {i//batch_size+1}: requested {len(batch)} IDs → got {len(data)} results")

    for game in data:
        records.append({
            "universe_id": game.get("id"),
            "root_place_id": game.get("rootPlaceId"),
            "name": game.get("name"),
            "creator": game.get("creator", {}).get("name"),
            "genre": game.get("genre"),
            "visits": game.get("visits"),
            "playing": game.get("playing"),
            "favorites": game.get("favoritedCount"),
            "max_players": game.get("maxPlayers"),
            "created": game.get("created"),
            "updated": game.get("updated")
        })
    time.sleep(0.5)

df = pd.DataFrame(records)
df.to_csv("roblox_games.csv", index=False, encoding="utf-8-sig")
print("✅ Saved roblox_games.csv with", len(df), "rows")
