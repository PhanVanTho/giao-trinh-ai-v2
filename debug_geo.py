import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def tim_kiem_tieu_de(lang: str, tu_khoa: str, gioi_han: int = 10):
    print(f"Searching '{tu_khoa}' on {lang}.wikipedia.org with limit {gioi_han}...", flush=True)
    try:
        r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params={
            "action": "query",
            "list": "search",
            "srsearch": tu_khoa,
            "srlimit": gioi_han,
            "format": "json",
        }, timeout=60, verify=False)
        data = r.json()
        search_results = data.get("query", {}).get("search", [])
        print(f"Found {len(search_results)} results for '{lang}':", flush=True)
        for i, item in enumerate(search_results):
            print(f"  {i+1}. {item['title']}")
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    print("--- TESTING GEOGRAPHY ---")
    tim_kiem_tieu_de("vi", "Địa lí", 5)
    tim_kiem_tieu_de("en", "Geography", 5)
    print("--- TESTING SPECIFIC TOPIC ---")
    tim_kiem_tieu_de("vi", "Địa lý học", 5)
    tim_kiem_tieu_de("en", "Physical Geography", 5)
