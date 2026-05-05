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
        # ★ここで好きなパスワードを設定してください
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

# パスワード認証
if not check_password():
    st.stop()

# ==========================================
# 2. データ取得と予測モデル（裏側の計算）
# ==========================================
@st.cache_data(ttl=3600) 
def run_prediction_model():
    # 米国セクターETF（予測のヒントにする銘柄）
    us_tickers = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLI', 'XLB']
    
    # 日本の業種別ETF（TOPIX-17シリーズ）全17銘柄
    jp_tickers = {
        '1617.T': '食品', '1618.T': 'エネルギー・資源', '1619.T': '建設・資材',
        '1620.T': '素材・化学', '1621.T': '医薬品', '1622.T': '自動車・輸送機',
        '1623.T': '鉄鋼・非鉄', '1624.T': '機械', '1625.T': '電機・精密',
        '1626.T': '情報通信・サービス他', '1627.T': '電力・ガス', '1628.T': '運輸・物流',
        '1629.T': '商社・卸売', '1630.T': '小売', '1631.T': '銀行',
        '1632.T': '金融（除く銀行）', '1633.T': '不動産'
    }
    
    # 直近のデータを取得（過去90日分）
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    
    try:
        # データのダウンロード
        us_raw = yf.download(us_tickers, start=start_date, end=end_date)['Close']
        jp_raw = yf.download(list(jp_tickers.keys()), start=start_date, end=end_date)['Close']
        
        # リターン（変化率）に変換
        us_returns = us_raw.pct_change().dropna()
        jp_returns = jp_raw.pct_change().dropna()
        
        # 行を合わせる（前日の米国で今日の日本を予測するため、データを1日ずらす）
        common_index = us_returns.index.intersection(jp_returns.index)
        us_final = us_returns.loc[common_index].iloc[:-1] # 最新日の1日前まで
        jp_final = jp_returns.loc[common_index].iloc[1:]  # 翌日のデータ
        
        # 最小行数でカット
        min_len = min(len(us_final), len(jp_final))
        us_train = us_final.iloc[-min_len:]
        jp_train = jp_final.iloc[-min_len:]
        
        # 【部分空間正則化PCAの近似】
        # 1. Ledoit-Wolf法で共分散行列を安定化
        lw = LedoitWolf().fit(us_train)
        # 2. PCAで主成分（市場ファクター）を抽出
        pca = PCA(n_components=3)
        pca.fit(us_train)
        
        # 回帰分析（米国ファクターが日本業種に与える影響度を計算）
        us_factors = pca.transform(us_train)
        beta = np.linalg.pinv(us_factors.T @ us_factors) @ us_factors.T @ jp_train.values
        
        # 【予測実行】
        # 最新の（昨晩の）米国市場リターンを使って、今日の日本を予測
        latest_us_return = us_returns.iloc[-1].values.reshape(1, -1)
        latest_factors = pca.transform(latest_us_return)
        pred_returns = latest_factors @ beta
        
        # 結果の整理
        pred_scores = pred_returns.flatten()
        results = []
        for i, (ticker, name) in enumerate(jp_tickers.items()):
            results.append({
                "銘柄コード": ticker.replace('.T', ''),
                "業種名": name,
                "予測スコア": round(pred_scores[i] * 100, 4)
            })
            
        df_results = pd.DataFrame(results).sort_values(by="予測スコア", ascending=False).reset_index(drop=True)
        return df_results, True
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return pd.DataFrame(), False

# ==========================================
# 3. Web画面の表示
# ==========================================
st.set_page_config(page_title="日米業種リードラグ予測", layout="wide")

st.title("📊 日米業種リードラグ 予測ダッシュボード")
st.write(f"**取得日時:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.info("米国セクターの動きから、本日の日本市場（日中）の強弱を予測します。")

with st.spinner("計算中..."):
    df_ranking, success = run_prediction_model()

if success:
    # トレンドアイコンの設定
    def get_icon(score):
        if score > 0.3: return "☀️ (強気)"
        elif score > 0: return "🌤️ (やや強気)"
        elif score > -0.3: return "☁️ (やや弱気)"
        else: return "☔️ (弱気)"

    df_ranking.insert(0, "トレンド", df_ranking["予測スコア"].apply(get_icon))

    # ハイライト表示
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"### ☀️ 本日の買い推奨\n**{df_ranking.iloc[0]['銘柄コード']} ({df_ranking.iloc[0]['業種名']})**")
    with col2:
        st.error(f"### ☔️ 本日の売り推奨\n**{df_ranking.iloc[-1]['銘柄コード']} ({df_ranking.iloc[-1]['業種名']})**")

    st.divider()

    # 全ランキング表示
    st.subheader("🏆 全17業種 予測ランキング")
    st.dataframe(df_ranking, use_container_width=True, height=650)
    
    st.caption("※本ツールは特定の銘柄の売買を推奨するものではありません。投資は自己責任でお願いします。")
else:
    st.warning("データの取得に失敗しました。市場が休場、または通信エラーの可能性があります。")
