import calendar
import csv
import time
import pandas as pd
import requests
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.config import config

API_KEY = config.get_secret("NEWS_API_KEY")
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')


def fetch_nyt_articles(query, begin_date, end_date, page=0):
    """
    NYT Article Search API로 기사를 가져오는 함수
    - query: 문자열(검색어/쿼리)
    - begin_date, end_date: YYYYMMDD 형식
    - page: 페이지 번호(0~99)
    """
    params = {
        "q": query,
        "begin_date": begin_date,
        "end_date": end_date,
        "sort": "oldest",  # or 'newest'
        "page": page,
        "api-key": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data


def get_articles_for_quarter(keyword, year, quarter):
    """
    특정 키워드(keyword)에 대해 특정 분기(quarter)의 기사를 수집
    """
    quarter_map = {
        1: (1, 3),   # 1월 ~ 3월
        2: (4, 6),   # 4월 ~ 6월
        3: (7, 9),   # 7월 ~ 9월
        4: (10, 12), # 10월 ~ 12월
    }
    # 분기 시작일 (year, start_month, 1)
    start_month, end_month = quarter_map[quarter]
    begin_date = f"{year}{start_month:02d}01"

    # 분기 종료일: end_month의 마지막 날
    last_day = calendar.monthrange(year, end_month)[1]
    end_date = f"{year}{end_month:02d}{last_day:02d}"

    print(f"[Keyword: {keyword} | Year: {year} | Q{quarter} | {begin_date} ~ {end_date}]")

    all_articles = []
    for page in range(100):
        data = fetch_nyt_articles(keyword, begin_date, end_date, page)

        # 응답이 정상인지 확인
        if "response" not in data or "docs" not in data["response"]:
            break

        docs = data["response"]["docs"]
        if not docs:
            # 더 이상 기사가 없으면 반복 중단
            break

        print(f" - Page {page + 1}: {len(docs)} articles")
        # 기사 정보 추출
        for doc in docs:
            headline = doc["headline"]["main"] if "headline" in doc else ""
            pub_date = doc["pub_date"] if "pub_date" in doc else ""
            web_url  = doc["web_url"] if "web_url" in doc else ""
            snippet  = doc.get("snippet", "")

            all_articles.append({
                "keyword": keyword,
                "year": year,
                "quarter": quarter,
                "headline": headline,
                "pub_date": pub_date,
                "web_url": web_url,
                "snippet": snippet
            })

        # 너무 빠른 요청 → rate limit 문제가 될 수 있으므로 약간의 지연
        time.sleep(20)

    return all_articles

def get_articles_for_year(keyword, year):
    """
    다중 키워드(keywords_list)에 대해,
    start_year ~ end_year 범위를 분기(1~4) 단위(begin_date, end_date)로 나누어 기사를 수집합니다.
    """
    # 분기별 (시작 월, 종료 월) 매핑
    

    all_articles = []


    for q in range(1, 5):
        articles = get_articles_for_quarter(keyword, year, q)
        all_articles.extend(articles)
        time.sleep(1)
    
    return all_articles


def get_articles_for_period(keywords_list, start_year, end_year):
    """
    특정 기간 동안의 특정 키워드(keyword)에 대한 기사를 수집
    """
    all_articles = []
    filename = f"inflation_news_{start_year}_{end_year}.csv"
    fieldnames = ["keyword", "year", "quarter", "headline", "pub_date", "web_url", "snippet"]

    if os.path.exists(filename):
        print(f"이미 존재하는 파일: {filename}")
        reader = csv.DictReader(open(filename, "r", encoding="utf-8"))
        all_articles = [row for row in reader]
        return all_articles


    with open(filename, "w", newline='', encoding="utf-8") as f:
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for year in range(start_year, end_year + 1):
            for keyword in keywords_list:
                articles_data = get_articles_for_year(keyword, year)
                print(f"총 수집 기사 수: {len(articles_data)}")
                for article in articles_data:
                    writer.writerow(article)
    
    return all_articles

def get_finbert_continuous_score(text):
    """
    문장(text)에 대해 FinBERT 로짓을 직접 계산,
    p(positive), p(negative)를 구해 (p(pos) - p(neg)) 형태의 [-1,1] 점수를 반환
    """
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert(**inputs)
        logits = outputs.logits  # shape: (batch_size=1, 3)
    probs = torch.softmax(logits, dim=1)[0]  # 첫 번째 배치
    p_neutral  = probs[0].item()  # label=0
    p_positive = probs[1].item()  # label=1
    p_negative = probs[2].item()  # label=2

    # 점수: (양성 확률 - 음성 확률)
    score = p_positive - p_negative
    return score

def analyze_articles(articles_data):
    analyzed_data = []
    for article in articles_data:
        # 우선 snippet을 확인
        snippet = article.get("snippet", "").strip()
        headline = article.get("headline", "").strip()

        # snippet이 비어있지 않으면 snippet 사용, 그렇지 않으면 headline 사용
        if snippet:
            text_to_analyze = snippet
        elif headline:
            text_to_analyze = headline
        else:
            # snippet과 headline 둘 다 빈 경우 => 0점 처리
            article["sentiment_score"] = 0.0
            analyzed_data.append(article)
            continue

        # 실제 분석 수행
        score = get_finbert_continuous_score(text_to_analyze)
        article["sentiment_score"] = score
        
        analyzed_data.append(article)

    return analyzed_data

def merge_semantic_score(articles_data):
    """
    기사의 sentiment 점수의 평균을 통해 일별 시계열 데이터로 변환
    """
    
    # 일별 기사 수집
    daily_data = {}
    for article in articles_data:
        pub_date = article["pub_date"]
        score = article["sentiment_score"]
        date = pub_date[:10]  # YYYY-MM-DD 형식
        if date not in daily_data:
            daily_data[date] = {"count": 0, "total_score": 0.0}
        daily_data[date]["count"] += 1
        daily_data[date]["total_score"] += score

    # 일별 평균 계산
    daily_scores = []
    for date, data in daily_data.items():
        avg_score = data["total_score"] / data["count"]
        daily_scores.append({
            "Date": date,
            "avg_score": avg_score,
            "count": data["count"]
        })

    return daily_scores

def save_merged_articles(articles_data, filename):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        fieldnames = ["Date", "avg_score", "count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in articles_data:
            writer.writerow(row)

def get_news_semantic_score(START_DATE:str, END_DATE:str, news_data_path:str) -> pd.DataFrame:
    # 뉴스 데이터 로드
    
    if os.path.exists(news_data_path):
        articles = pd.read_csv(news_data_path)

        # 기간 필터링
        articles = articles[articles["Date"] >= START_DATE]
        articles = articles[articles["Date"] <= END_DATE]

        return articles
    else:
        raise FileNotFoundError(f"존재하지 않는 파일: {news_data_path}")


    
if __name__ == "__main__":
    keywords_list = [
        "inflation",
    ]
    start_year = 2018
    end_year = 2024

    articles = get_articles_for_period(keywords_list, start_year, end_year)

    # 기사 분석
    analyzed_data = analyze_articles(articles)

    merged_data = merge_semantic_score(analyzed_data)

    save_merged_articles(merged_data, "news.csv")

    