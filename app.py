import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import datetime

# ==========================================
# 1. ログイン認証（パスワード保護）
# ==========================================
def check_password():
    """簡単なパスワード認証を行います"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        # ★ここで好きなパスワードを設定してください（現在は「test2026」）
        if st.session_state["password_input"] == "test2026":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 ログイン")
        st.text_input("パスワードを入力してください", type="password", key="password_input", on_change=password_entered)
        if st.session_state.get("password_input") and not st.session_state["password_correct"]:
            st.error("パスワードが間違っています。")
        return False
    return True

# パスワードが間違っている場合はここで画面をストップ
if not check_password():
    st.stop()


# ==========================================
# 2. データ取得と予測モデル（裏側の計算）
# ==========================================
# @st.cache_data をつけると、毎回計算せず結果を一時保存して高速化します
@st.cache_data(ttl=3600) 
def run_prediction_model():
    # 対象とするETFのティッカー（米国のセクターETFと日本の業種ETF）
    us_tickers = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY'] # 米国：情報技術、金融、ヘルスケア、エネルギー、一般消費財
    jp_tickers = {'1625.T': '電機・精密', '1631.T': '銀行', '1626.T': '情報通信', '1622.T': '自動車', '1629.T': '商社'}
    
    # 直近60日間のデータを取得
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    
    try:
        # 米国ETFの前日終値データを取得
        us_data = yf.download(us_tickers, start=start_date, end=end_date)['Close'].pct_change().dropna()
        # 日本ETFの終値データを取得
        jp_data = yf.download(list(jp_tickers.keys()), start=start_date, end=end_date)['Close'].pct_change().dropna()
        
        # モデルの学習（部分空間正則化PCAの近似：Ledoit-Wolf + PCA）
        cov_estimator = LedoitWolf()
        pca = PCA(n_components=2)
        
        # 学習（最新日以外を使用）
        train_us = us_data.iloc[:-1]
        train_jp = jp_data.iloc[1:] # 翌日の日本を予測するため1日ずらす
        
        # 行数を合わせる
        min_len = min(len(train_us), len(train_jp))
        train_us = train_us.iloc[-min_len:]
        train_jp = train_jp.iloc[-min_len:]
        
        cov_estimator.fit(train_us)
        pca.fit(train_us)
        us_factors = pca.transform(train_us)
        
        # 感応度（ベータ）の計算
        beta = np.linalg.pinv(us_factors.T @ us_factors) @ us_factors.T @ train_jp.values
        
        # 予測（最新の米国データを使用）
        latest_us = us_data.iloc[-1].values.reshape(1, -1)
        latest_factors = pca.transform(latest_us)
        prediction = latest_factors @ beta
        
        # 予測結果をデータフレームにまとめる
        pred_scores = prediction.flatten()
        results = []
        for i, (ticker, name) in enumerate(jp_tickers.items()):
            results.append({
                "銘柄コード": ticker.replace('.T', ''),
                "業種": name,
                "予測スコア": round(pred_scores[i] * 100, 3) # 見やすくスケールアップ
            })
            
        df_results = pd.DataFrame(results).sort_values(by="予測スコア", ascending=False).reset_index(drop=True)
        return df_results, True
        
    except Exception as e:
        # 万が一Yahoo Financeからデータが取れなかった場合のエラー回避用ダミーデータ
        dummy = pd.DataFrame({
            "銘柄コード": ["1625", "1626", "1622", "1629", "1631"],
            "業種": ["電機・精密", "情報通信", "自動車", "商社", "銀行"],
            "予測スコア": [0.85, 0.62, 0.21, -0.58, -0.92]
        })
        return dummy, False


# ==========================================
# 3. Webダッシュボードの表示（表側の画面）
# ==========================================
st.title("📊 日米業種リードラグ 予測ダッシュボード")
today_str = datetime.date.today().strftime("%Y年%m月%d日")
st.write(f"**更新日:** {today_str} (米国市場 引け後データ適用)")

# 裏側の計算を実行
with st.spinner("最新の市場データを取得してAIモデルを計算中..."):
    df_ranking, is_real_data = run_prediction_model()

if not is_real_data:
    st.warning("⚠️ Yahoo Financeからのデータ取得に失敗したため、テスト用データを表示しています。")

# トレンドアイコンを付与
def get_trend_icon(score):
    if score >= 0.5: return "☀️ (強気)"
    elif score >= 0: return "🌤️ (やや強気)"
    elif score >= -0.5: return "☁️ (やや弱気)"
    else: return "☔️ (弱気)"

df_ranking.insert(0, "トレンド", df_ranking["予測スコア"].apply(get_trend_icon))

# サマリーの表示
st.subheader("💡 本日の推奨アクション")
col1, col2 = st.columns(2)
top_long = df_ranking.iloc[0]
top_short = df_ranking.iloc[-1]

with col1:
    st.success(f"**☀️ 買い推奨 (Long)**\n\n**{top_long['銘柄コード']}**: {top_long['業種']} \n\n(スコア: {top_long['予測スコア']})")
with col2:
    st.error(f"**☔️ 売り推奨 (Short)**\n\n**{top_short['銘柄コード']}**: {top_short['業種']} \n\n(スコア: {top_short['予測スコア']})")

# ランキング表の表示
st.subheader("🏆 全業種 ETF 予測ランキング")
st.dataframe(df_ranking, use_container_width=True)

st.caption("※このダッシュボードは「部分空間正則化付きPCAを用いた日米業種リードラグ戦略」のベースラインモデルに基づいています。投資判断は自己責任で行ってください。")
