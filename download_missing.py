import urllib.request
import os

os.makedirs("data/raw", exist_ok=True)

urls = {
    "nina34.txt": "https://psl.noaa.gov/data/correlation/nina34.data",
    "era5_wpwp.json": "https://archive-api.open-meteo.com/v1/archive?latitude=10.0&longitude=140.0&start_date=1996-01-01&end_date=2024-12-31&daily=temperature_2m_mean,precipitation_sum&timezone=UTC",
    "era5_bob.json": "https://archive-api.open-meteo.com/v1/archive?latitude=14.0&longitude=88.0&start_date=1996-01-01&end_date=2024-12-31&daily=temperature_2m_mean,precipitation_sum&timezone=UTC"
}

for filename, url in urls.items():
    print(f"Downloading {filename}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req, timeout=120) as response:
        with open(f"data/raw/{filename}", "wb") as f:
            f.write(response.read())
    print(f"Successfully saved to data/raw/{filename}")