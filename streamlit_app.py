import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sqlite3
import io

# --- 1. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('alpha_pro_v5.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios 
                 (name TEXT PRIMARY KEY, idx_t TEXT, eq_t TEXT, db_t TEXT, gd_t TEXT, 
                  capital REAL, stretch REAL, buffer REAL, slip REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS holdings 
                 (port_name TEXT PRIMARY KEY, cash REAL, eq_shares REAL, db_shares REAL, gd_shares REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS daily_stats 
                 (port_name TEXT, date TEXT, nav REAL, benchmark_px REAL, PRIMARY KEY (port_name, date))''')
    conn.commit()
    conn.close()

init_db()

# --- 2. CORE LOGIC ---
def calculate_signals(data, ticker_idx):
    price_col = data[ticker_idx]
    df = pd.DataFrame(price_col).rename(columns={ticker_idx: 'price'})
    df['ema50'] = df['price'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['price'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['price'].ewm(span=200, adjust=False).mean()
    df['below_all'] = (df['price'] < df['ema50']) & (df['price'] < df['ema100']) & (df['price'] < df['ema200'])
    df['above_50'] = (df['price'] > df['ema50'])
    return df

def get_target_alloc(row, history, stretch_threshold):
    p, e50, e100, e200 = row['price'], row['ema50'], row['ema100'], row['ema200']
    hist_week = history.resample('W-FRI').last()
    cons_above_2 = hist_week['above_50'].iloc[-2:].all() if len(hist_week) >= 2 else False
    
    if cons_above_2: return (0.95, 0.00, 0.05)
    if p > e50 and p > (e50 * (1 + stretch_threshold)): return (0.90, 0.05, 0.05)
    if not cons_above_2 and row['above_50']: return None
    
    cons_below_3 = hist_week['below_all'].iloc[-3:].all() if len(hist_week) >= 3 else False
    if cons_below_3: return (0.80, 0.10, 0.10)
    if p < e50 and p < e100 and p < e200: return (0.65, 0.15, 0.20)
    if p < e50 and p < e100: return (0.75, 0.10, 0.15)
    return (0.95, 0.00, 0.05)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Alpha Hedge Pro", layout="wide")
st.title("🛡️ Alpha Hedge Strategy")

with st.sidebar:
    st.header("⚙️ Configuration")
    idx_t = st.text_input("Index (Benchmark)", "^NSEI")
    eq_t = st.text_input("Equity ETF", "NIFTYBEES.NS")
    db_t = st.text_input("Debt ETF", "LICNETFGSC.NS")
    gd_t = st.text_input("Gold ETF", "GOLDBEES.NS")
    st.markdown("---")
    cap_input = st.number_input("Capital", 100000)
    str_input = st.slider("EMA Stretch %", 0.01, 0.10, 0.05)
    buf_input = st.number_input("Buffer %", 0.001, 0.01, 0.002, format="%.3f")
    slp_input = st.number_input("Slippage %", 0.0005, 0.005, 0.001, format="%.3f")

tickers_global = {'index': idx_t, 'equity': eq_t, 'debt': db_t, 'gold': gd_t}
tab1, tab2 = st.tabs(["📉 Analytical Backtest", "💰 Live Execution"])

# --- TAB 1: BACKTEST ---
with tab1:
    c1, c2, c3 = st.columns(3)
    ind_dt = c1.date_input("Indicator Start", datetime(2016, 1, 1))
    st_dt = c2.date_input("Backtest Start", datetime(2022, 1, 1))
    en_dt = c3.date_input("Backtest End", datetime.today())

    if st.button("🚀 Run Comprehensive Analysis"):
        with st.spinner("Fetching data..."):
            raw = yf.download(list(tickers_global.values()), start=ind_dt, end=en_dt)['Close'].ffill().dropna()
            
            signals = calculate_signals(raw, tickers_global['index'])
            etf_data = raw[[tickers_global['equity'], tickers_global['debt'], tickers_global['gold']]]
            
            cash, holdings = cap_input, {t: 0 for t in [tickers_global['equity'], tickers_global['debt'], tickers_global['gold']]}
            portfolio_log, trans_log, last_alloc = [], [], None
            
            for date, row in signals.iterrows():
                if date < pd.to_datetime(st_dt): continue
                curr_prices = etf_data.loc[date]
                mv = sum(holdings[t] * curr_prices[t] for t in holdings)
                total_val = mv + cash
                
                if date.weekday() == 4:
                    target_w = get_target_alloc(row, signals.loc[:date], str_input)
                    if target_w and target_w != last_alloc:
                        usable = total_val * (1 - buf_input)
                        for i, t in enumerate([tickers_global['equity'], tickers_global['debt'], tickers_global['gold']]):
                            holdings[t] = (usable * target_w[i]) // (curr_prices[t] * (1 + slp_input))
                            trans_log.append({'Date': date, 'Security': t, 'Shares': holdings[t], 'Price': curr_prices[t]})
                        cash = total_val - sum(holdings[t] * curr_prices[t] for t in holdings)
                        last_alloc = target_w
                
                portfolio_log.append({'Date': date, 'Portfolio_Value': total_val})

            port_df = pd.DataFrame(portfolio_log).set_index('Date')
            bench_series = raw[tickers_global['index']].loc[st_dt:]
            
            s_ret = port_df['Portfolio_Value'].pct_change().fillna(0)
            b_ret = bench_series.pct_change().fillna(0)
            s_cum = (1 + s_ret).cumprod()
            b_cum = (1 + b_ret).cumprod()
            
            dd_s = s_cum / s_cum.cummax() - 1
            dd_b = b_cum / b_cum.cummax() - 1
            
            w, rf = 30, 0.06 / 252
            roll_vol_s = s_ret.rolling(w).std() * np.sqrt(252)
            roll_vol_b = b_ret.rolling(w).std() * np.sqrt(252)
            roll_sharpe_s = (s_ret.rolling(w).mean() - rf) / s_ret.rolling(w).std() * np.sqrt(252)
            roll_sharpe_b = (b_ret.rolling(w).mean() - rf) / b_ret.rolling(w).std() * np.sqrt(252)
            roll_beta = s_ret.rolling(w).cov(b_ret) / b_ret.rolling(w).var()

            fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=("Growth", "Drawdown (%)", "30D Sharpe", "30D Vol", "30D Beta"))
            
            fig.add_trace(go.Scatter(x=s_cum.index, y=s_cum, name="Strategy", line=dict(color='#00CC96')), row=1, col=1)
            fig.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name="Bench", line=dict(color='#636EFA', dash='dot')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=dd_s.index, y=dd_s*100, fill='tozeroy', name="Strat DD", line=dict(color='red')), row=2, col=1)
            fig.add_trace(go.Scatter(x=dd_b.index, y=dd_b*100, name="Bench DD", line=dict(color='gray', dash='dot')), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=roll_sharpe_s.index, y=roll_sharpe_s, name="Strat Sharpe", line=dict(color='#00CC96')), row=3, col=1)
            fig.add_trace(go.Scatter(x=roll_sharpe_b.index, y=roll_sharpe_b, name="Bench Sharpe", line=dict(color='gray', dash='dot')), row=3, col=1)
            
            fig.add_trace(go.Scatter(x=roll_vol_s.index, y=roll_vol_s, name="Strat Vol", line=dict(color='#00CC96')), row=4, col=1)
            fig.add_trace(go.Scatter(x=roll_vol_b.index, y=roll_vol_b, name="Bench Vol", line=dict(color='gray', dash='dot')), row=4, col=1)
            
            fig.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta, name="30D Beta", line=dict(color='orange')), row=5, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=5, col=1)

            fig.update_layout(height=1400, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: LIVE EXECUTION & SCHEDULER ---
with tab2:
    conn = sqlite3.connect('alpha_pro_v5.db')
    existing = [r[0] for r in conn.execute("SELECT name FROM portfolios").fetchall()]
    
    sel = st.selectbox("Select Live Portfolio", ["+ Initialize New"] + existing)
    
    if sel == "+ Initialize New":
        name = st.text_input("Name this Portfolio")
        if st.button("Go Live"):
            conn.execute("INSERT OR REPLACE INTO portfolios VALUES (?,?,?,?,?,?,?,?,?)", 
                        (name, idx_t, eq_t, db_t, gd_t, cap_input, str_input, buf_input, slp_input))
            conn.execute("INSERT OR REPLACE INTO holdings VALUES (?,?,0,0,0)", (name, cap_input))
            conn.commit()
            st.rerun()
    else:
        p = conn.execute("SELECT * FROM portfolios WHERE name=?", (sel,)).fetchone()
        h = conn.execute("SELECT * FROM holdings WHERE port_name=?", (sel,)).fetchone()
        
        # 1. DELETE OPTION
        if st.sidebar.button(f"🗑️ Delete {sel}"):
            conn.execute("DELETE FROM portfolios WHERE name=?", (sel,))
            conn.execute("DELETE FROM holdings WHERE port_name=?", (sel,))
            conn.execute("DELETE FROM daily_stats WHERE port_name=?", (sel,))
            conn.commit()
            st.rerun()

        # 2. DATA SYNC
        port_tickers = [p[1], p[2], p[3], p[4]] 
        with st.spinner(f"Updating {sel} with latest market data..."):
            live_data = yf.download(port_tickers, period="1mo", interval="1d")['Close'].ffill()
            px_eq, px_db, px_gd = live_data[p[2]].iloc[-1], live_data[p[3]].iloc[-1], live_data[p[4]].iloc[-1]
            idx_px = live_data[p[1]].iloc[-1]
            
            total_nav = h[1] + (h[2]*px_eq) + (h[3]*px_db) + (h[4]*px_gd)
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            conn.execute("INSERT OR IGNORE INTO daily_stats VALUES (?, ?, ?, ?)", (sel, today_str, total_nav, idx_px))
            conn.commit()

        # 3. LIVE GRAPH
        st.subheader(f"📊 {sel} Real-Time Growth")
        hist_data = pd.read_sql(f"SELECT * FROM daily_stats WHERE port_name='{sel}' ORDER BY date", conn)
        
        if len(hist_data) > 1:
            hist_data['strat_growth'] = hist_data['nav'] / hist_data['nav'].iloc[0]
            hist_data['bench_growth'] = hist_data['benchmark_px'] / hist_data['benchmark_px'].iloc[0]
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(x=hist_data['date'], y=hist_data['strat_growth'], name="Portfolio", line=dict(color='#00CC96')))
            fig_mini.add_trace(go.Scatter(x=hist_data['date'], y=hist_data['bench_growth'], name="Index", line=dict(color='gray', dash='dot')))
            fig_mini.update_layout(height=350, template="plotly_white", margin=dict(t=20, b=20))
            st.plotly_chart(fig_mini, use_container_width=True)

        

        c1, c2, c3 = st.columns(3)
        c1.metric("Current NAV", f"₹{total_nav:,.2f}")
        c2.metric("Net Profit", f"₹{(total_nav-p[5]):,.2f}", f"{((total_nav/p[5])-1)*100:.2f}%")
        c3.metric("Latest Index", f"{idx_px:,.2f}")

        # 4. TRADE MANAGER (REAL-TIME REBALANCE)
        st.markdown("---")
        st.subheader("🚨 Trade Manager")
        
        sig = calculate_signals(live_data, p[1])
        target = get_target_alloc(sig.iloc[-1], sig, p[6])
        
        if target:
            usable_cap = total_nav * (1 - p[7])
            p_map = {p[2]: px_eq, p[3]: px_db, p[4]: px_gd}
            assets = [p[2], p[3], p[4]]
            
            t_sh = [(usable_cap * w) // (p_map[t] * (1 + p[8])) for w, t in zip(target, assets)]
            cur_sh = [h[2], h[3], h[4]]
            trades = [ts - cs for ts, cs in zip(t_sh, cur_sh)]
            
            if any(t != 0 for t in trades):
                st.warning("Strategy signal changed! Trades required to stay in alignment.")
                trade_df = pd.DataFrame({
                    'Asset': assets,
                    'Action': ['BUY' if t > 0 else 'SELL' if t < 0 else 'HOLD' for t in trades],
                    'Qty': [abs(int(t)) for t in trades],
                    'LTP': [round(p_map[t], 2) for t in assets]
                })
                st.table(trade_df)
                
                if st.button("Confirm & Rebalance Portfolio"):
                    new_cash = total_nav - sum(ts * p_map[t] for ts, t in zip(t_sh, assets))
                    conn.execute("UPDATE holdings SET cash=?, eq_shares=?, db_shares=?, gd_shares=? WHERE port_name=?", 
                                (new_cash, t_sh[0], t_sh[1], t_sh[2], sel))
                    conn.commit()
                    st.success("Rebalance Complete!")
                    st.rerun()
            else:
                st.success("✅ Portfolio is optimized for the latest market data. No trades needed.")
        else:
            st.info("Waiting for next signal... Market is in a neutral zone.")
    conn.close()