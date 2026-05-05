import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import datetime
import urllib.parse

# --- 1. ログイン認証 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    def password_entered():
        if st.session_state["password_input"] == "test2026":
            st.session_state["password_correct"] = True
    if not st.session_state["password_correct"]:
        st.title("🔒 ログイン")
        st.text_input("パスワード", type="password", key="password_input", on_change=password_entered)
        return False
    return True

if not check_password():
    st.stop()

# --- 2. 計算ロジック ---
@st.cache_data(ttl=3600)
def run_full_analysis():
    # 米国セクター（表示名付き）
    us_sectors = {
        'XLK': '情報技術', 'XLF': '金融', 'XLV': 'ヘルスケア', 
        'XLE': 'エネルギー', 'XLY': '一般消費財', 'XLI': '資本財', 'XLB': '素材'
    }
    # 日本業種
    jp_tickers = {
        '1617.T': '食品', '1618.T': 'エネルギー・資源', '1619.T': '建設・資材',
        '1620.T': '素材・化学', '1621.T': '医薬品', '1622.T': '自動車・輸送機',
        '1623.T': '鉄鋼・非鉄', '1624.T': '機械', '1625.T': '電機・精密',
        '1626.T': '情報通信・サービス他', '1627.T': '電力・ガス', '1628.T': '運輸・物流',
        '1629.T': '商社・卸売', '1630.T': '小売', '1631.T': '銀行',
        '1632.T': '金融（除く銀行）', '1633.T': '不動産'
    }
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    
    try:
        # データ取得
        us_data = yf.download(list(us_sectors.keys()), start=start_date, end=end_date)['Close']
        jp_data = yf.download(list(jp_tickers.keys()), start=start_date, end=end_date)['Close']
        
        # 前日比計算
        us_ret = us_data.pct_change().dropna()
        jp_ret = jp_data.pct_change().dropna()
        
        # 直近の米国騰落（理由として表示するため）
        latest_us_perf = us_ret.iloc[-1] * 100
        us_perf_df = pd.DataFrame({
            'セクター': [us_sectors[t] for t in latest_us_perf.index],
            '騰落率(%)': latest_us_perf.values.round(2)
        }).sort_values('騰落率(%)', ascending=False)

        # モデル計算（PCA）
        common = us_ret.index.intersection(jp_ret.index)
        us_f = us_ret.loc[common].iloc[:-1]
        jp_f = jp_ret.loc[common].iloc[1:]
        min_l = min(len(us_f), len(jp_f))
        
        pca = PCA(n_components=3).fit(us_f.iloc[-min_l:])
        us_factors = pca.transform(us_f.iloc[-min_l:])
        beta = np.linalg.pinv(us_factors.T @ us_factors) @ us_factors.T @ jp_f.iloc[-min_l:].values
        
        # 予測
        pred = pca.transform(us_ret.iloc[-1:].values) @ beta
        
        res = []
        for i, (t, name) in enumerate(jp_tickers.items()):
            res.append({"銘柄コード": t.replace('.T',''), "業種名": name, "予測スコア": round(pred[0][i]*100, 4)})
        
        return pd.DataFrame(res).sort_values("予測スコア", ascending=False), us_perf_df, True
    except:
        return pd.DataFrame(), pd.DataFrame(), False

# --- 3. 画面表示 ---
st.set_page_config(page_title="日米リードラグ分析", layout="wide")
st.title("📊 日米業種リードラグ予測 & 理由分析")

df, us_df, success = run_full_analysis()

if success:
    # --- サマリーエリア ---
    col1, col2 = st.columns(2)
    with col1:
        top = df.iloc[0]
        st.success(f"### ☀️ 今日の買い：{top['業種名']} ({top['銘柄コード']})")
        q_long = urllib.parse.quote(f"日本株 {top['業種名']} ニュース")
        st.markdown(f"[🔗 この業種のニュースをチェック](https://www.google.com/search?q={q_long}&tbm=nws)")
        
    with col2:
        bottom = df.iloc[-1]
        st.error(f"### ☔️ 今日の売り：{bottom['業種名']} ({bottom['銘柄コード']})")
        q_short = urllib.parse.quote(f"日本株 {bottom['業種名']} ニュース")
        st.markdown(f"[🔗 この業種のニュースをチェック](https://www.google.com/search?q={q_short}&tbm=nws)")

    st.divider()

    # --- 理由の分析エリア ---
    st.subheader("🧐 なぜこの予測になったのか？（米国市場の振り返り）")
    st.write("昨晩の米国市場のセクター別騰落状況です。この動きが「理由」となって日本の予測スコアが計算されています。")
    
    # 米国騰落を横並びで表示
    us_cols = st.columns(len(us_df))
    for i, row in enumerate(us_df.itertuples()):
        color = "red" if row._2 < 0 else "green"
        us_cols[i].metric(row.セクター, f"{row._2}%")

    st.divider()

    # --- 全ランキング ---
    st.subheader("🏆 全17業種 予測ランキング")
    def get_icon(s):
        if s > 0.3: return "☀️"
        elif s > 0: return "🌤️"
        elif s > -0.3: return "☁️"
        else: return "☔️"
    df.insert(0, "トレンド", df["予測スコア"].apply(get_icon))
    st.dataframe(df, use_container_width=True, height=600)

else:
    st.error("データ取得エラー。時間をおいて試してください。")
