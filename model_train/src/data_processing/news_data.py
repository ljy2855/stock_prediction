from time import sleep
import requests
import json
import os
from datetime import datetime, timedelta

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.config import config

API_KEY = config.get_secret("NEWS_API_KEY")

TRUSTED_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",  # Financial Times
    "cnbc.com",
    "forbes.com",
    "marketwatch.com",
    "nytimes.com",  # The New York Times
    "economist.com",
    "cnn.com"
]

def build_query(keywords, logic="AND"):
    """
    여러 키워드를 논리 연산자로 연결하여 검색 쿼리를 생성합니다.
    :param keywords: 키워드 리스트
    :param logic: 논리 연산자 (AND, OR, NOT 중 하나)
    :return: 조합된 검색 쿼리
    """
    if not keywords or logic not in {"AND", "OR", "NOT"}:
        raise ValueError("키워드와 논리 연산자를 확인하세요.")
    return f" {logic} ".join([f'"{keyword}"' for keyword in keywords])


def fetch_news(query, from_date, to_date, language="en", page_size=100):
    """
    NewsAPI를 사용하여 뉴스 데이터를 가져옵니다.
    :param query: 검색 키워드 (예: "FED")
    :param from_date: 시작 날짜 (ISO 형식, 예: "2023-01-01")
    :param to_date: 종료 날짜 (ISO 형식, 예: "2023-01-31")
    :param language: 뉴스 언어 (기본값: "en")
    :param page_size: 페이지 크기 (최대: 100)
    :return: 뉴스 데이터 리스트
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "domains": ",".join(TRUSTED_DOMAINS),
        "to": to_date,
        "language": language,
        "pageSize": page_size,
        "sortBy": "relevancy",
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

def split_date_range(start_date, end_date, interval_days=30):
    """
    시작 날짜와 종료 날짜를 interval_days 단위로 나눕니다.
    :param start_date: 시작 날짜 (datetime 객체)
    :param end_date: 종료 날짜 (datetime 객체)
    :param interval_days: 나눌 간격 (일 단위, 기본값: 30일)
    :return: 날짜 범위 리스트 (튜플 형식)
    """
    date_ranges = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=interval_days), end_date)
        date_ranges.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end + timedelta(days=1)
    return date_ranges

def save_news(news_data, output_path):
    """
    뉴스를 JSON 파일로 저장합니다.
    :param news_data: 뉴스 데이터 리스트
    :param output_path: 저장 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(news_data, f, indent=4)
    print(f"News saved to {output_path}")


def download_news_data(keywords,start_date,end_date):
    # 날짜 범위를 한 달 단위로 나눕니다.
    date_ranges = split_date_range(start_date, end_date, interval_days=30)

    all_news = []

    query = build_query(keywords, logic="OR")
    
    print(f"Fetching news for keyword: {query}")
    for from_date, to_date in date_ranges:
        print(f"Fetching news from {from_date} to {to_date}...")
        news = fetch_news(query, from_date, to_date)
        all_news.extend(news)
        sleep(1)

    save_news(all_news, "data/raw/news/sp500_news.json")
    return all_news

if __name__ == "__main__":
    # 검색 키워드 및 날짜
    keywords = ["FED", "S&P 500"]
    start_date = datetime(2024, 11, 19)  # 시작 날짜 (API 제공하는게 최근 한달만 가능)
    end_date = datetime.today()  # 오늘까지

    download_news_data(keywords, start_date, end_date)
