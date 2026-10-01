"""
ui/styles.py — CSS kept from v3 (citation badges, hover popups, metric bars).

Changes: colours use the active Streamlit theme where possible, and popups
also open on TAP / keyboard focus (`:focus-within`), because phones have no
hover — v3 popups were unusable on mobile.
"""

CSS = """
<style>
  .block-container{padding-top:1.2rem;}
  .answer-box{
    background:rgba(127,127,127,.06);border:1px solid rgba(127,127,127,.25);border-radius:10px;
    padding:1rem 1.2rem;font-size:.97rem;line-height:1.85;position:relative;overflow-wrap:anywhere;}
  .answer-box ul{margin:.2rem 0 .2rem 1.1rem;padding:0;}
  .cite-wrap{position:relative;display:inline-block;outline:none;}
  .cite-badge{
    display:inline-block;background:#1f4e6b;color:#cfe8ff;border:1px solid #2d6e9e;border-radius:4px;
    padding:0 6px;font-size:.72rem;font-weight:700;cursor:pointer;user-select:none;vertical-align:middle;margin:0 2px;}
  .cite-badge.bad{background:#5a1e1e;border-color:#a33;color:#ffd7d7;}
  .cite-wrap:hover .cite-badge,.cite-wrap:focus-within .cite-badge{background:#2d6e9e;}
  .cite-popup{
    display:none;position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);
    width:min(520px,88vw);background:#161b22;border:1px solid #388bfd;border-radius:8px;padding:10px 14px;
    z-index:9999;box-shadow:0 8px 32px rgba(0,0,0,.6);font-size:.8rem;line-height:1.5;color:#e6edf3;text-align:left;}
  .cite-popup::after{content:"";position:absolute;top:100%;left:50%;transform:translateX(-50%);
    border:7px solid transparent;border-top-color:#388bfd;}
  .cite-wrap:hover .cite-popup,.cite-wrap:focus-within .cite-popup{display:block;}
  .pop-header{font-weight:700;color:#58a6ff;font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;
    border-bottom:1px solid #30363d;padding-bottom:4px;margin-bottom:5px;}
  .pop-doc{color:#e6edf3;font-size:.76rem;margin-bottom:2px;font-weight:600;}
  .pop-section{color:#8b949e;font-size:.72rem;margin-bottom:3px;}
  .pop-breadcrumb{color:#8b949e;font-size:.7rem;font-style:italic;margin-bottom:5px;}
  .pop-meta-row{margin-bottom:6px;}
  .pop-meta-chip{display:inline-block;background:#21262d;color:#79c0ff;border-radius:3px;padding:1px 6px;
    margin:0 4px 2px 0;font-size:.68rem;}
  .pop-body{font-family:ui-monospace,monospace;font-size:.76rem;white-space:pre-wrap;word-break:break-word;
    background:#0d1117;border-radius:4px;padding:6px 8px;max-height:200px;overflow-y:auto;color:#adbac7;
    border-left:3px solid #388bfd;}
  .src-body{font-family:ui-monospace,monospace;font-size:.8rem;white-space:pre-wrap;word-break:break-word;
    background:rgba(127,127,127,.08);border-radius:6px;padding:.6rem .8rem;border-left:3px solid #58a6ff;}
  .rqs-badge{display:inline-block;border:1px solid;border-radius:8px;padding:.35rem 1rem;margin-bottom:.6rem;}
  .metric-cell{text-align:center;}
  .metric-name{font-size:.72rem;opacity:.75;margin-bottom:3px;}
  .metric-val{font-size:1.2rem;font-weight:700;}
  .metric-bar{background:rgba(127,127,127,.25);border-radius:4px;height:6px;margin-top:4px;}
  .metric-fill{height:6px;border-radius:4px;}
  .metric-tip{font-size:.64rem;opacity:.6;margin-top:3px;}
  .nf-box{border:1px dashed #d29922;border-radius:10px;padding:.8rem 1.1rem;background:rgba(210,153,34,.08);}
</style>
"""
