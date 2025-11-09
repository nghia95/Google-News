
import pandas as pd
from GoogleNews import GoogleNews # GoogleNewsなどの必要なライブラリもインポート
import json
from google import genai
from google.genai.errors import APIError

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fetch_stock_news_from_google_news(query:str):
    """Retrieves recent news articles related to a specific financial or market query from Google News.

    Args:
        query (str): The keyword or phrase for the search (e.g., "Toyota stock forecast", "Nikkei 225 news").
        
    Returns:
        dict: A Data dictionary containing the collected news data (e.g., Title, Summary, Source, URL).
    """
    print(f"✅ 検索キーワード: '{query}' でGoogle Newsを検索中...")

    # GoogleNewsオブジェクトを初期化
    googlenews = GoogleNews(lang='en', region='US') # 日本語、日本リージョンで設定
    # googlenews = GoogleNews(lang='ja', region='ja') # 日本語、日本リージョンで設定
    # 検索クエリを設定し、指定ページ数まで結果を取得
    googlenews.search(query)
    
    # 結果を取得
    results = googlenews.results()
    
    news_data = {} # 辞書型として初期化
    article_index = 1 # 記事のキーとして使用するカウンター

    for item in results:
        # 必要な情報を抽出し、一時的なディクショナリ（記事データ）を作成
        article_dict = {
            'title': item.get('title'),
            'source': item.get('publisher'),
            'date': item.get('date'),
            'url': item.get('link'),
            'summary': item.get('desc')
        }
    
        # 💡 修正点: .append() の代わりに、連番をキーとしてディクショナリに保存
        news_data[f'article_{article_index}'] = article_dict
        
        # 次のキーのためにカウンターを増やす
        article_index += 1
    
    if not news_data:
        print("❌ 該当するニュース記事が見つかりませんでした。")
        return {
            "status": "failed",
            "count": 0,
            "message": "No news articles found for the query.",
            "articles": []
        }

    print(f"✅ 合計 {len(news_data)} 件の記事が見つかりました。")
    print("#######################",news_data)
    # ファイルに書き込む
    with open("./data.json", 'w', encoding='utf-8') as f:
        # indent=4 で整形して書き込むと、ファイルが読みやすくなります
        json.dump(news_data, f)

    return {"status": "OK"}


def predict_index(target_index: str) -> dict:
    """The minimum function required to analyze news articles and have Gemini generate a prediction and its rationale."""
    # ----------------------------------------------------
    # 💡 注意: clientは通常モジュールレベルで一度だけ初期化します
    # ファイルを読み込む
    with open("./data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("######data",data)
    try:
        client = genai.Client()
        print("Gemini Clientの初期化しました")
    except Exception as e:
        # 環境変数または認証情報に問題がある可能性があります
        print(f"Gemini Clientの初期化に失敗しました: {e}")
        client = None
    # ----------------------------------------------------
    # 1. 最小限の初期チェック
    if client is None:
        return {"status": "error", "message": "Client not initialized."}

    # if not query:
    #     return {"status": "data_insufficient", "message": "No news data provided."}

    
    # 3. 最小限のプロンプトと必須JSON形式
    prompt = f"""
    Based *only* on the provided news articles in the {data}, analyze the sentiment and predict the closing direction for the **{target_index}** index. 
    Output the result strictly in the required JSON format.

    Required JSON Format:
    {{
      "predicted_close": <float or current index value>,
      "market_sentiment": "<Bullish|Bearish|Neutral>",
      "analysis_basis": "<Concise summary of the market drivers, max 50 words>"
    }}
    """
    
    # 4. API呼び出しと最小限のエラー処理
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        
        # JSON応答をパース
        llm_analysis = json.loads(response.text)
        
        # 💡 デバッグ用: 応答が成功した場合、Geminiの解析結果をそのまま返して構造を確認
        return {
            "prediction_status": "success",
            "target_index": target_index,
            "llm_output": llm_analysis # 解析結果全体を返して確認
        }

    except Exception as e:
        # ネットワーク、認証、JSONパースなど、全てのエラーをキャッチ
        return {
            "prediction_status": "runtime_error",
            "message": f"An unexpected error occurred during API call or JSON parsing: {type(e).__name__}: {str(e)}",
            "error_type": type(e).__name__
        }
    
