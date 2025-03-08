import os
import json
import time
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import requests

BING_API_KEY = os.getenv("BING_API_KEY", "TU_CLAVE_DE_BING")
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

def bing_search(query):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": 1}
    resp = requests.get(BING_ENDPOINT, headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if "webPages" in data and "value" in data["webPages"]:
            return data["webPages"]["value"][0].get("snippet", "")
    return ""

def scrape_categories():
    print("Iniciando scraping de categorías...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        })

        URL = "https://www.webtenerife.com/que-hacer/"
        page.goto(URL)
        page.wait_for_load_state("networkidle")

        elements = page.query_selector_all("div.card__col article.card")
        print(f"Categorías encontradas: {len(elements)}")

        categories = []

        for item in elements:
            try:
                category_name = item.query_selector("h6.card__title.heading").inner_text().strip()
            except:
                category_name = "Sin nombre"

            try:
                description = item.query_selector("div.card__description").inner_text().strip()
            except:
                description = "Sin descripción"

            try:
                link = item.query_selector("a").get_attribute("href")
                if not link.startswith("http"):
                    link = urljoin(URL, link)
            except:
                link = "Sin enlace"

            # Usamos Bing para obtener una "descripción detallada" tomando el link como query
            descripcion_detallada = bing_search(link)

            categories.append({
                "nombre": category_name,
                "descripcion": description,
                "link": link,
                "descripcion_detallada": descripcion_detallada
            })

            print(f"Categoría: {category_name}")
            print(f"Descripción: {description}")
            print(f"Link: {link}")
            print(f"Descripción detallada (Bing): {descripcion_detallada}")
            print("-----")

        with open("categorias_tenerife.json", "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=4)

        print("Scraping completado y datos guardados en categorias_tenerife.json")

if __name__ == "__main__":
    scrape_categories()