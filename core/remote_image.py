import requests

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"

# Wikimedia's API rejects requests with no/generic User-Agent (403), per
# their API etiquette policy: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "face-recognition-poc/1.0 (internal technical validation)"


def resolve_commons_file_url(file_title: str) -> str:

    response = requests.get(
        COMMONS_API_URL,
        params={
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    pages = response.json()["query"]["pages"]

    page = next(iter(pages.values()))
    if "missing" in page or "imageinfo" not in page:
        raise ValueError(f"Commons file not found: {file_title}")

    return page["imageinfo"][0]["url"]


def download_image(url: str) -> bytes:
    """Plain HTTP GET of an image URL. No auth needed, but a descriptive
    User-Agent avoids Wikimedia's anti-bot 403s (see USER_AGENT above)."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.content
