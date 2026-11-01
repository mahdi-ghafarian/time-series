from datetime import date, timedelta
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st
import pickle
from pathlib import Path

STARTUP_RESULT_FILE = Path(__file__).parent / "startup_result.pkl"

st.set_page_config(page_title='Exit Strategy', page_icon='📊', layout='wide')

DEFAULT_STOPS = [1,2,3,4,5,7.5,10,12.5,15,20,25,30,35,40,45,50,-1]
DEFAULT_TARGETS = [1,2,3,4,5,7.5,10,12.5,15,20,25,30,35,40,45,50,100,-1]

@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query):
    quotes = yf.Search(query, max_results=8, news_count=0, lists_count=0, include_cb=False).quotes
    return [{'symbol': str(q['symbol']), 'name': str(q.get('longname') or q.get('shortname') or q['symbol']),
             'exchange': str(q.get('exchDisp') or q.get('exchange') or '')} for q in quotes if q.get('symbol')]

def select_ticker(symbol):
    st.session_state['bt_ticker'] = symbol.upper()
    st.session_state['bt_search'] = ''

@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker, start_date, end_date, interval):
    # yfinance end is exclusive, so add one day to make the UI's end date inclusive.
    yf_end = end_date + timedelta(days=1)
    data = yf.download(ticker, start=start_date.isoformat(), end=yf_end.isoformat(), interval=interval,
                       auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError('No data downloaded. Check the ticker, dates, and interval.')
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)
    required = ['Open','High','Low','Close']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    data = data[required].dropna().sort_index()
    if len(data) < 2:
        raise ValueError('At least two valid price bars are required.')
    if (data[required] <= 0).any().any():
        raise ValueError('Price data contains zero or negative prices.')
    return data

def select_random_entries(data, n, seed):
    rng = np.random.default_rng(seed)
    n = min(n, len(data)-1)
    idx = np.sort(rng.choice(np.arange(len(data)-1), size=n, replace=False))
    return pd.DataFrame({'EntryIndex': idx, 'EntryDate': data.index[idx], 'EntryClose': data['Close'].to_numpy()[idx]})

def simulate_trade(data, entry_index, stop_loss_pct, target_pct, commission_pct, slippage_pct, conservative):
    close_prices=data['Close'].to_numpy(); high_prices=data['High'].to_numpy(); low_prices=data['Low'].to_numpy(); dates=data.index
    entry_ref=float(close_prices[entry_index]); no_stop=stop_loss_pct == -1; no_target=target_pct == -1
    entry_exec=entry_ref*(1+slippage_pct/100)
    stop_price=None if no_stop else entry_ref*(1-float(stop_loss_pct)/100)
    target_price=None if no_target else entry_ref*(1+float(target_pct)/100)
    exit_index=len(data)-1; exit_ref=float(close_prices[exit_index]); reason='EndOfData'
    for i in range(entry_index+1, len(data)):
        stop_hit=stop_price is not None and float(low_prices[i]) <= stop_price
        target_hit=target_price is not None and float(high_prices[i]) >= target_price
        if stop_hit and target_hit:
            exit_ref=stop_price if conservative else target_price; reason='Stop' if conservative else 'Target'; exit_index=i; break
        if stop_hit: exit_ref=stop_price; reason='Stop'; exit_index=i; break
        if target_hit: exit_ref=target_price; reason='Target'; exit_index=i; break
    exit_exec=exit_ref*(1-slippage_pct/100)
    gross=(exit_exec/entry_exec-1)*100; total_commission=2*commission_pct; net=gross-total_commission
    risk=100 if no_stop else float(stop_loss_pct)
    return {'EntryDate':dates[entry_index], 'ExitDate':dates[exit_index], 'EntryIndex':entry_index, 'ExitIndex':exit_index,
            'EntryReferencePrice':entry_ref, 'EntryExecutionPrice':entry_exec, 'ExitReferencePrice':exit_ref,
            'ExitExecutionPrice':exit_exec, 'StopLossPct':stop_loss_pct, 'TargetPct':target_pct,
            'GrossReturnPct':gross, 'NetReturnPct':net, 'RealizedR':net/risk, 'BarsHeld':exit_index-entry_index,
            'ExitReason':reason, 'CommissionPct':total_commission}

def calculate_metrics(trades, stop, target):
    r=trades['NetReturnPct'].to_numpy(float); wins=r>0; losses=r<0; wr=r[wins]; lr=r[losses]
    win_rate=float(wins.mean()*100) if len(r) else np.nan
    avg_win=float(wr.mean()) if len(wr) else np.nan; avg_loss=float(lr.mean()) if len(lr) else np.nan
    expectancy=float(r.mean()) if len(r) else np.nan; risk=100 if stop == -1 else float(stop); er=expectancy/risk
    planned=np.nan if target == -1 else float(target)/risk
    realized=avg_win/abs(avg_loss) if len(wr) and len(lr) and avg_loss != 0 else np.nan
    be=abs(avg_loss)/(avg_win+abs(avg_loss))*100 if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_win>0 and avg_loss<0 else np.nan
    pbe=1/(1+planned)*100 if np.isfinite(planned) and planned>0 else np.nan
    counts=trades['ExitReason'].value_counts(); total=len(trades)
    return {'StopLossPct':stop,'TargetPct':target,'NumberOfTrades':total,'WinRatePct':win_rate,'AverageWinPct':avg_win,
            'AverageLossPct':avg_loss,'RealizedRRR':realized,'BreakEvenWinRatePct':be,'PlannedRRR':planned,
            'PlannedBreakEvenWinRatePct':pbe,'ExpectancyPct':expectancy,'ExpectancyR':er,
            'AverageBarsHeld':float(trades['BarsHeld'].mean()) if total else np.nan,
            'StopExitPct':counts.get('Stop',0)/total*100 if total else np.nan,
            'TargetExitPct':counts.get('Target',0)/total*100 if total else np.nan,
            'EndOfDataExitPct':counts.get('EndOfData',0)/total*100 if total else np.nan,
            'AverageCommissionPct':float(trades['CommissionPct'].mean()) if total else np.nan}

def compare_combinations(data, entries, stops, targets, commission, slippage, conservative, rank_metric):
    results=[]; details=[]; total=len(stops)*len(targets); done=0
    progress=st.progress(0, text='Testing combinations...')
    for stop in stops:
        for target in targets:
            combo=[simulate_trade(data,int(e.EntryIndex),stop,target,commission,slippage,conservative) for e in entries.itertuples(index=False)]
            df=pd.DataFrame(combo); results.append(calculate_metrics(df,stop,target)); details.extend(combo); done += 1
            progress.progress(done/total, text=f'Testing combinations: {done}/{total}')
    progress.empty(); rdf=pd.DataFrame(results); tdf=pd.DataFrame(details)
    for metric in ['ExpectancyR','WinRatePct']:
        sd=rdf[metric].std(); rdf[f'z_{metric}']=(rdf[metric]-rdf[metric].mean())/sd if sd and np.isfinite(sd) else 0.0
    rdf['ScoreRaw']=2*rdf['z_ExpectancyR']+rdf['z_WinRatePct']; lo=rdf['ScoreRaw'].min(); hi=rdf['ScoreRaw'].max()
    rdf['Score']=100*(rdf['ScoreRaw']-lo)/(hi-lo) if hi != lo else 50.0
    rdf=rdf.sort_values(rank_metric,ascending=False,na_position='last').reset_index(drop=True); rdf.insert(0,'Rank',np.arange(1,len(rdf)+1))
    return rdf,tdf

def create_heatmap(results, stops, targets, ticker, start_date, end_date, n_entries, top_metric, bottom_metric, color_metric, vmin, vmax, wr_min, er_min):
    color=results.pivot(index='TargetPct',columns='StopLossPct',values=color_metric).reindex(index=targets,columns=stops)
    top=results.pivot(index='TargetPct',columns='StopLossPct',values=top_metric).reindex(index=targets,columns=stops)
    bottom=results.pivot(index='TargetPct',columns='StopLossPct',values=bottom_metric).reindex(index=targets,columns=stops)
    cond=pd.concat([results[['TargetPct','StopLossPct']],((results['WinRatePct']>wr_min)&(results['ExpectancyR']>er_min)).rename('Condition')],axis=1).pivot(index='TargetPct',columns='StopLossPct',values='Condition').reindex(index=targets,columns=stops)
    vals=color.to_numpy(float); finite=vals[np.isfinite(vals)]
    if not finite.size: return None
    vmin=float(np.min(finite)) if vmin is None else vmin; vmax=float(np.max(finite)) if vmax is None else vmax
    if vmin==vmax: pad=max(abs(vmin)*0.01,1e-9); vmin-=pad; vmax+=pad
    fig,ax=plt.subplots(figsize=(max(12,len(stops)*.85),max(8,len(targets)*.55))); image=ax.imshow(vals,cmap='RdYlGn',aspect='auto',interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    fig.colorbar(image,ax=ax).set_label(color_metric); ax.set_title(f'{ticker} | {start_date} to {end_date} | {n_entries:,} Random Entries\n{top_metric} (top), {bottom_metric} (bottom)')
    ax.set_xlabel('Stop-Loss (%)'); ax.set_ylabel('Target (%)'); ax.set_xticks(np.arange(len(stops))); ax.set_xticklabels(['NS' if x==-1 else str(x) for x in stops]); ax.set_yticks(np.arange(len(targets))); ax.set_yticklabels(['NT' if x==-1 else str(x) for x in targets])
    formats={'ExpectancyR':'.2f','ExpectancyPct':'.2f','WinRatePct':'.1f','RealizedRRR':'.2f','Score':'.0f'}
    tf=formats.get(top_metric,'.2f'); bf=formats.get(bottom_metric,'.2f')
    for row in range(len(targets)):
        for col in range(len(stops)):
            a=top.iloc[row,col]; b=bottom.iloc[row,col]; cv=vals[row,col]
            label=f'{a:{tf}}\n{b:{bf}}' if np.isfinite(a) and np.isfinite(b) else 'N/A'
            rgba=image.cmap(image.norm(cv)) if np.isfinite(cv) else (1,1,1,1); brightness=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
            ax.text(col,row,label,ha='center',va='center',color='black' if brightness>.55 else 'white',fontsize=8)
            if bool(cond.iloc[row,col]): ax.add_patch(Rectangle((col-.5,row-.5),1,1,fill=False,edgecolor='black',linewidth=2))
    ax.set_xticks(np.arange(-.5,len(stops),1),minor=True); ax.set_yticks(np.arange(-.5,len(targets),1),minor=True); ax.grid(which='minor',color='gray',linewidth=.4,alpha=.4); ax.tick_params(which='minor',bottom=False,left=False); fig.tight_layout(); return fig

st.title('Exit Strategy')
st.write('Test stop-loss and target combinations across randomly selected historical entries.')
with st.sidebar:
    st.header('Data')
    q=st.text_input('Search for ticker',placeholder='Type a ticker or company name',key='bt_search',live='500ms').strip()
    if len(q)>=2:
        try:
            for m in search_tickers(q):
                st.button(f"{m['symbol']} | {m['name']} | {m['exchange'] or 'Unknown exchange'}",key=f"srch_{m['symbol']}",on_click=select_ticker,args=(m['symbol'],),width='stretch')
        except Exception as exc: st.warning(f'Yahoo Finance ticker search failed: {exc}')
    ticker=st.text_input('Ticker',value='QQQ',key='bt_ticker').strip().upper()
    interval=st.selectbox('Interval',['1d','1wk','1mo'],index=0)
    st.subheader('Date range')
    start_date=st.date_input('Start date',value=date(2010,1,1))
    end_date=st.date_input('End date',value=date(2026,10,30))
    st.divider(); st.header('Simulation')
    n_entries=st.number_input('Random entries',min_value=1,value=1000,step=100)
    seed=st.number_input('Random seed',min_value=0,value=27,step=1)
    commission=st.number_input('Commission (%)',min_value=0.0,value=0.0,step=.01,format='%.3f')
    slippage=st.number_input('Slippage (%)',min_value=0.0,value=0.0,step=.01,format='%.3f')
    conservative=st.checkbox('Stop first when stop and target hit same candle',value=True)
    st.divider(); st.header('Heatmap')
    METRICS=['StopLossPct','TargetPct','NumberOfTrades','WinRatePct','AverageWinPct','AverageLossPct','RealizedRRR',
             'BreakEvenWinRatePct','PlannedRRR','PlannedBreakEvenWinRatePct','ExpectancyPct','ExpectancyR','AverageBarsHeld',
             'StopExitPct','TargetExitPct','EndOfDataExitPct','AverageCommissionPct','Score']
    hm_top=st.selectbox('Top metric',METRICS,index=METRICS.index('ExpectancyR'))
    hm_bottom=st.selectbox('Bottom metric',METRICS,index=METRICS.index('WinRatePct'))
    hm_color=st.selectbox('Color metric',METRICS,index=METRICS.index('ExpectancyR'))
    auto_scale=st.checkbox('Auto color scale (use min/max)',value=False)
    c1,c2=st.columns(2)
    hm_vmin=c1.number_input('Color min',value=0.0,step=0.1,disabled=auto_scale)
    hm_vmax=c2.number_input('Color max',value=1.0,step=0.1,disabled=auto_scale)
    if auto_scale: hm_vmin=hm_vmax=None
    c3,c4=st.columns(2)
    hl_wr=c3.number_input('Highlight WR min',value=50.0,step=1.0)
    hl_er=c4.number_input('Highlight ExpR min',value=0.5,step=0.1)
    rank_metric=st.selectbox('Rank metric',METRICS,index=METRICS.index('ExpectancyR'))
    st.divider(); st.header('Combinations')
    with st.expander('Stop / target values'):
        stop_text=st.text_area('Stops (%)', ', '.join(map(str,DEFAULT_STOPS)), help='Use -1 for no stop (NS).')
        target_text=st.text_area('Targets (%)', ', '.join(map(str,DEFAULT_TARGETS)), help='Use -1 for no target (NT).')
    st.divider()
    run_simulation=st.button('Run simulation',type='primary',width='stretch')

def parse_values(s): return [float(x.strip()) for x in s.split(',') if x.strip()]

if start_date >= end_date: st.warning('The start date must be earlier than the end date.'); st.stop()
if not ticker: st.warning('Enter a ticker symbol.'); st.stop()
def run_simulation_now(
    ticker, start_date, end_date, interval, n_entries, seed,
    commission, slippage, conservative, stops, targets,
):
    with st.spinner(f"Loading {ticker} data..."):
        data = download_data(ticker, start_date, end_date, interval)

    entries = select_random_entries(data, int(n_entries), int(seed))
    t0 = time.perf_counter()

    results, trades = compare_combinations(
        data, entries, stops, targets, float(commission),
        float(slippage), conservative, "ExpectancyR",
    )

    return (
        data, entries, results, trades, stops, targets,
        time.perf_counter() - t0, ticker, start_date, end_date,
        int(n_entries),
    )


def load_startup_result():
    if not STARTUP_RESULT_FILE.exists():
        return None
    try:
        with open(STARTUP_RESULT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        st.warning(f"The startup result could not be loaded: {exc}")
        return None


def save_startup_result(output):
    # This is the only result file the app creates. User simulations are
    # kept only in that user's Streamlit session.
    try:
        with open(STARTUP_RESULT_FILE, "wb") as f:
            pickle.dump(output, f)
    except OSError as exc:
        st.warning(
            "The initial result is displayed, but this server did not allow "
            f"startup_result.pkl to be written: {exc}"
        )


# Explicit button: editing controls never starts a simulation.
if run_simulation:
    try:
        stops = parse_values(stop_text)
        targets = parse_values(target_text)
        if not stops or not targets:
            raise ValueError("Provide at least one stop and one target value.")

        st.session_state["bt_output"] = run_simulation_now(
            ticker, start_date, end_date, interval, n_entries, seed,
            commission, slippage, conservative, stops, targets,
        )
        st.caption(
            "Showing this session's latest simulation. It is not written "
            "to the server filesystem."
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

# First load for a user: show the one shared startup result. If the file does
# not exist yet, calculate the current defaults once and create that one file.
elif "bt_output" not in st.session_state:
    startup_output = load_startup_result()

    if startup_output is not None:
        st.session_state["bt_output"] = startup_output
        st.caption("Showing the precomputed startup result.")
    else:
        try:
            stops = parse_values(stop_text)
            targets = parse_values(target_text)
            st.info(
                "No startup result exists yet. Running the initial simulation "
                "once and creating startup_result.pkl."
            )
            startup_output = run_simulation_now(
                ticker, start_date, end_date, interval, n_entries, seed,
                commission, slippage, conservative, stops, targets,
            )
            save_startup_result(startup_output)
            st.session_state["bt_output"] = startup_output
        except Exception as exc:
            st.error(str(exc))
            st.stop()
else:
    st.caption(
        "Change parameters as needed, then click Run simulation. "
        "The displayed result remains unchanged until you run again."
    )

if 'bt_output' in st.session_state:
    data,entries,results,trades,stops,targets,elapsed,out_ticker,out_start,out_end,out_n=st.session_state['bt_output']
    c=st.columns(4); c[0].metric('Bars',f'{len(data):,}'); c[1].metric('Random entries',f'{len(entries):,}'); c[2].metric('Combinations',f'{len(results):,}'); c[3].metric('Run time',f'{elapsed:.1f}s')
    st.subheader('Heatmap')
    fig=create_heatmap(results,stops,targets,out_ticker,out_start,out_end,len(entries),hm_top,hm_bottom,hm_color,hm_vmin,hm_vmax,hl_wr,hl_er)
    if fig is not None: st.pyplot(fig,width='stretch'); plt.close(fig)
    ranked=results.drop(columns='Rank').sort_values(rank_metric,ascending=False,na_position='last').reset_index(drop=True)
    ranked.insert(0,'Rank',np.arange(1,len(ranked)+1))
    display_cols=['Rank','StopLossPct','TargetPct','NumberOfTrades','ExpectancyR','ExpectancyPct','WinRatePct','AverageWinPct','AverageLossPct','RealizedRRR','StopExitPct','TargetExitPct','EndOfDataExitPct','Score']
    with st.expander(f'Ranked combinations (by {rank_metric})',expanded=False):
        st.dataframe(ranked[display_cols].round(2),hide_index=True,width='stretch',height=600)
    with st.expander('Random entry bars'): st.dataframe(entries,hide_index=True,width='stretch')
    with st.expander('Trade details'): st.dataframe(trades.round(4),hide_index=True,width='stretch',height=600)
    st.subheader('Download')
    dl=st.columns(3)
    dl[0].download_button('Results CSV',ranked.round(4).to_csv(index=False).encode(),f'{out_ticker.lower()}_stop_target_results.csv','text/csv',width='stretch')
    dl[1].download_button('Trade details CSV',trades.round(4).to_csv(index=False).encode(),f'{out_ticker.lower()}_trade_details.csv','text/csv',width='stretch')
    dl[2].download_button('Random entries CSV',entries.to_csv(index=False).encode(),f'{out_ticker.lower()}_random_entries.csv','text/csv',width='stretch')
