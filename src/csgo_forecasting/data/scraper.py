"""Data scraping utilities for HLTV.org"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import random
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc


def load_page_soup(url: str, sleep_time: int = 3) -> BeautifulSoup:
    """
    Load and parse a web page using Selenium and BeautifulSoup.
    
    Args:
        url: URL to load
        sleep_time: Base time to wait (will add random noise)
        
    Returns:
        BeautifulSoup object containing parsed HTML
    """
    print(f"\rGetting {url}", end="")
    
    try:
        options = uc.ChromeOptions()
        # options.add_argument("--headless=new")  # New headless mode
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Specify Chrome binary location for WSL
        options.binary_location = "/usr/bin/google-chrome"  # Or try "/usr/bin/chromium-browser"
        
        driver = uc.Chrome(options=options, use_subprocess=True)
        driver.get(url)
        
        # Random sleep to mimic human behavior
        time.sleep(random.uniform(2, 4))

        # Vérifier si on a la vraie page
        page_source = driver.page_source
        if "challenge-success-text" in page_source or "Checking your browser" in page_source:
            print(" [Still blocked, waiting more...]", end="")
            time.sleep(15)
            page_source = driver.page_source
        
        soup = BeautifulSoup(driver.page_source, "lxml")
        driver.quit()
        
        return soup
        
    except Exception as e:
        print(f"\nError: {e}, retrying in {sleep_time}s...")
        time.sleep(sleep_time + random.uniform(0, 2))
        return load_page_soup(url, sleep_time)


def get_rating_trend(url: str, sleep_time: int = 3, max_retries: int = 3) -> List[Dict]:
    """
    Extract rating trend data from player page.
    
    Args:
        url: Player page URL
        sleep_time: Base time to sleep between requests
        max_retries: Maximum number of retries before giving up
        
    Returns:
        List of dictionaries containing rating trend data
    """
    retries = 0
    
    while retries < max_retries:
        soup = load_page_soup(url, sleep_time)
        
        # Debug: Afficher les divs trouvés
        print(f"\n=== DEBUG pour {url} ===")
        
        # Chercher les divs avec uniqueChart
        perf_trend = soup.findAll("div", id=lambda x: x and x.startswith("uniqueChart"))
        print(f"Nombre de divs 'uniqueChart' trouvés: {len(perf_trend)}")
        
        if perf_trend:
            print(f"Premier div trouvé - ID: {perf_trend[0].get('id')}")
            print(f"Attributs du div: {perf_trend[0].attrs}")
        
        # Chercher tous les divs avec un ID
        all_divs_with_id = soup.findAll("div", id=True)
        print(f"\nTotal de divs avec ID: {len(all_divs_with_id)}")
        print("IDs trouvés (20 premiers):")
        for div in all_divs_with_id[:20]:
            print(f"  - {div.get('id')}")
        
        # Chercher des patterns alternatifs
        chart_divs = soup.findAll("div", class_=lambda x: x and "chart" in str(x).lower())
        print(f"\nDivs avec 'chart' dans la classe: {len(chart_divs)}")
        for div in chart_divs[:5]:
            print(f"  - Classes: {div.get('class')}, ID: {div.get('id')}")
        
        # Chercher dans les scripts
        scripts = soup.findAll("script")
        print(f"\nScripts trouvés: {len(scripts)}")
        for script in scripts:
            if script.string and ("rating" in script.string.lower() or "chart" in script.string.lower()):
                print(f"Script avec 'rating' ou 'chart' trouvé (premiers 300 chars):")
                print(script.string[:300])
                break
        
        print("=== FIN DEBUG ===\n")
        
        try:
            perf_trend = json.loads(perf_trend[0]["data-fusionchart-config"])
            time.sleep(sleep_time + random.uniform(0, 2))
            return perf_trend["dataSource"]["data"]
            
        except (IndexError, KeyError) as e:
            retries += 1
            print(f"\nError parsing trend data (attempt {retries}/{max_retries}): {e}")
            
            if retries < max_retries:
                time.sleep(sleep_time + random.uniform(1, 3))
            else:
                print(f"\nCouldn't find trend data for {url}, returning empty list")
                return []


def get_individual_stats(url: str, sleep_time: int = 3) -> Dict:
    """
    Extract individual player statistics.
    
    Args:
        url: Player stats URL
        sleep_time: Base time to sleep between requests
        
    Returns:
        Dictionary of player statistics
    """
    soup = load_page_soup(url, sleep_time)
    data = soup.find_all("div", attrs={"class": "stats-row"}, limit=None)
    
    try:
        player_data = {}
        
        for stats_row in data:
            stats_row_str = str(stats_row)
            sub_handler = BeautifulSoup(stats_row_str, "html.parser")
            occurences = sub_handler.find_all("span")
            
            key = occurences[0].get_text()
            value = occurences[-1].get_text()
            player_data[key] = value
        
        time.sleep(sleep_time + random.uniform(0, 2))
        return player_data
        
    except Exception as e:
        print(f"\nError parsing stats for {url}: {e}, retrying")
        time.sleep(sleep_time + random.uniform(1, 3))
        return get_individual_stats(url, sleep_time)


def scrape_player_data(
    metadata_path: str,
    output_path: str,
    sleep_time: int = 3,
    selection: str = "all"
) -> pd.DataFrame:
    """
    Scrape complete player data from HLTV.
    
    Args:
        metadata_path: Path to player metadata JSON
        output_path: Path to save scraped data
        sleep_time: Base time to sleep between requests (random noise added)
        selection: Type of players to scrape ("all", "top10", etc.)
        
    Returns:
        DataFrame containing scraped player data
    """
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    data = pd.DataFrame(metadata["data"])
    data["index"] = range(len(data))
    data = data.set_index("index")
    
    # Scrape individual stats
    print("\nScraping individual stats...")
    for index, player in data.iterrows():
        # Random delay between players
        time.sleep(random.uniform(sleep_time, sleep_time + 3))
        
        # Modify URLs to access individual stats
        url_parts = player.url.split("/")[2:]
        url_parts.insert(3, "individual")
        indv_url = "http://" + "/".join(url_parts)
        
        # Get stats
        player_stats = get_individual_stats(indv_url, sleep_time)
        
        # Set player ID
        data.loc[index, "id"] = int(player.url.split("/")[5])
        
        # Set stats
        for key, value in player_stats.items():
            data.loc[index, key] = value
    
    # Scrape rating trends
    print("\nScraping rating trends...")
    data["rating_trend"] = data.url.apply(
        lambda url: get_rating_trend(url, sleep_time)
    )
    
    # Save data
    data = data.set_index("id")
    data.to_json(output_path)
    print(f"\nData saved to {output_path}")
    
    return data