# -*- coding: utf-8 -*-
"""
Bangkok Skipjack Proxy + 선행변수(FAD/유가/어획proxy) + SARIMAX 3개월 예측 + 차트 + 이메일(CID)

[필수 Secrets]
- EMAIL_ADDR, EMAIL_PASSWORD, RECEIVER_ADDR
- (권장) GOOGLE_API_KEY, GOOGLE_CSE_CX  -> 최신 INFOFISH ITN PDF 자동 검색

[옵션 우회 스위치(강력)]
- ITN_PDF_URL : CSE가 400/403 등으로 실패할 때, 이 URL이 있으면 "검색 없이" 바로 PDF 사용
  예) https://v4.infofish.org/media/attachments/2025/07/08/final--itn-6-2025_updated.pdf
"""

import os, re, io, ssl, datetime as dt
import requests
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


# =========================
# ENV
# =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX", "")

# (옵션) CSE 실패 시 이 URL로 바로 실행
ITN_PDF_URL = os.getenv("ITN_PDF_URL", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
EMAIL_ADDR = os.getenv("EMAIL_ADDR", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
RECEIVER_ADDR = os.getenv("RECEIVER_ADDR", "")

HISTORY_CSV = os.getenv("HISTORY_CSV", "bangkok_proxy_monthly.csv")

SEARCH_QUERY = 'site:infofish.org (itn OR "INFOFISH Trade News") pdf skipjack Thailand "US$"'
FRED_BRENT_MONTHLY_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILBRENTEU"

POS_PATTERNS = [r"strong catches", r"improved catches", r"higher catches", r"good catches"]
NEG_PATTERNS = [r"poor catches", r"low catches", r"weaker catches", r"declining catches"]


# =========================
# Utils
# =========================
def month_start_kst():
    kst = dt.datetime.utcnow() + dt.timedelta(hours=9)
    return pd.Timestamp(dt.date(kst.year, kst.month, 1))


def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def extract_text_first_pages(pdf_bytes: bytes, max_pages=15) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(max_pages, len(pdf.pages))
        for i in range(n):
            text += "\n" + (pdf.pages[i].extract_text() or "")
    return re.sub(r"\s+", " ", text)


# =========================
# Google CSE (에러 로그 강화)
# =========================
def google_cse_top_pdf_url(query: str) -> str:
    if not (GOOGLE_API_KEY and GOOGLE_CSE_CX):
        raise RuntimeError("GOOGLE_API_KEY / GOOGLE_CSE_CX 설정 필요")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": 5}
    r = requests.get(url, params=params, timeout=30)

    # ★ 400/403 등 실패 시, 원인(JSON)을 그대로 로그에 출력
    if not r.ok:
        print("=== Google CSE Request Failed ===")
        print("Status:", r.status_code)
        try:
            print("Body(JSON):", r.json())
        except Exception:
            print("Body(Text):", r.text)
        print("Tip: cx 값에 'cx=' 포함/공백/줄바꿈이 있거나, Custom Search API Enable 안 됐을 가능성이 큼")
        r.raise_for_status()

    data = r.json()
    for item in data.get("items", []):
        link = item.get("link", "")
        if link.lower().endswith(".pdf") and "infofish" in link.lower():
            return link

    raise RuntimeError("CSE 검색 결과에 PDF가 없습니다. 검색어/엔진 설정 또는 사이트 제한을 확인하세요.")


def resolve_itn_pdf_url() -> str:
    """
    1) ITN_PDF_URL 있으면 그걸 사용(검색 없이 바로)
    2) 없으면 Google CSE로 검색
    """
    if ITN_PDF_URL:
        return ITN_PDF_URL

    # CSE 실패하면 우회 스위치를 안내
    try:
        return google_cse_top_pdf_url(SEARCH_QUERY)
    except Exception as e:
        raise RuntimeError(
            "Google CSE로 ITN PDF 검색 실패.\n"
            "해결 1) Google Cloud에서 'Custom Search API' Enable 확인\n"
            "해결 2) GOOGLE_CSE_CX 값이 'cx=' 없이 값만 들어갔는지 확인\n"
            "임시 우회) ITN_PDF_URL 환경변수(Secret)로 최신 ITN PDF URL을 넣으면 검색 없이 실행됩니다.\n"
            f"원인: {e}"
        )


# =========================
# Extract: Price + Catch proxy
# =========================
def extract_skipjack_thailand_price(text: str) -> dict:
    # range: US$ 1,400-1,450/MT
    p_range = re.compile(
        r"(delivery price of frozen skipjack.*?Thailand.*?US\$\s*([\d,]{3,5})\s*[-–]\s*([\d,]{3,5})\s*/\s*MT)",
        re.IGNORECASE
    )
    m = p_range.search(text)
    if m:
        low = int(m.group(2).replace(",", ""))
        high = int(m.group(3).replace(",", ""))
        return {"low": low, "high": high, "sentence": m.group(1)}

    # single: US$ 1,650/MT
    p_single = re.compile(
        r"(delivery price of frozen skipjack.*?Thailand.*?US\$\s*([\d,]{3,5})\s*/\s*MT)",
        re.IGNORECASE
    )
    m = p_single.search(text)
    if m:
        val = int(m.group(2).replace(",", ""))
        return {"low": val, "high": val, "sentence": m.group(1)}

    # loose backup
    p_loose = re.compile(r"(skipjack.*?Thailand.*?US\$\s*[\d,]{3,5}.*?/MT)", re.IGNORECASE)
    m = p_loose.search(text)
    if m:
        sent = m.group(1)
        nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]{3,5}", sent)]
        if len(nums) >= 2 and ("-" in sent or "–" in sent):
            return {"low": min(nums), "high": max(nums), "sentence": sent}
        if len(nums) >= 1:
            return {"low": nums[0], "high": nums[0], "sentence": sent}

    raise RuntimeError("가격 문장 매칭 실패(패턴 업데이트 필요).")


def calc_catch_score(text: str) -> float:
    t = text.lower()
    pos = sum(len(re.findall(p, t)) for p in POS_PATTERNS)
    neg = sum(len(re.findall(p, t)) for p in NEG_PATTERNS)
    return float(pos - neg)


# =========================
# Brent (FRED)
# =========================
def load_brent_monthly() -> pd.DataFrame:
    df = pd.read_csv(FRED_BRENT_MONTHLY_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.rename(columns={"DATE": "month", "MCOILBRENTEU": "brent"})
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    df["brent"] = pd.to_numeric(df["brent"], errors="coerce")
    df = df.dropna(subset=["brent"])
    return df[["month", "brent"]]


# =========================
# FAD flag (완전 자동 룰)
# =========================
def fad_flag(month_ts: pd.Timestamp) -> int:
    # 기본 룰: 7~8월=1
    return 1 if int(month_ts.month) in (7, 8) else 0


# =========================
# History CSV
# =========================
def load_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
        return df.sort_values("month")
    return pd.DataFrame(columns=["month","price_mid","price_low","price_high","catch_score","pdf_url"])


def save_history(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"]).dt.strftime("%Y-%m-01")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def upsert_history(df: pd.DataFrame, month_ts: pd.Timestamp, low: int, high: int, catch_score: float, pdf_url: str) -> pd.DataFrame:
    mid = round((low + high) / 2, 2)
    row = {
        "month": month_ts,
        "price_mid": mid,
        "price_low": low,
        "price_high": high,
        "catch_score": round(float(catch_score), 3),
        "pdf_url": pdf_url
    }
    df = df[df["month"] != month_ts].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df.sort_values("month")


# =========================
# Train + Forecast
# =========================
def build_train(hist: pd.DataFrame) -> pd.DataFrame:
    brent = load_brent_monthly()
    df = hist.merge(brent, on="month", how="left")
    df["brent"] = df["brent"].ffill().bfill()
    df["fad"] = df["month"].apply(fad_flag).astype(int)
    df = df.sort_values("month").set_index("month")
    df = df.dropna(subset=["price_mid","brent","catch_score","fad"])
    return df


def sarimax_forecast_3m(train: pd.DataFrame) -> pd.DataFrame:
    y = train["price_mid"]
    X = train[["fad","brent","catch_score"]]

    model = SARIMAX(
        y, exog=X,
        order=(1,1,1),
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    last_month = train.index.max()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=3, freq="MS")

    last_brent = float(train["brent"].iloc[-1])
    last_catch = float(train["catch_score"].iloc[-1])

    future_exog = pd.DataFrame(index=future_months)
    future_exog["fad"] = [fad_flag(m) for m in future_months]
    future_exog["brent"] = last_brent
    future_exog["catch_score"] = last_catch

    fc = res.get_forecast(steps=3, exog=future_exog)
    ci = fc.conf_int()

    return pd.DataFrame({
        "month": [d.strftime("%Y-%m") for d in future_months],
        "forecast": fc.predicted_mean.values.round(2),
        "low_95": ci.iloc[:,0].values.round(2),
        "high_95": ci.iloc[:,1].values.round(2),
    })


# =========================
# Chart + Email (CID)
# =========================
def make_chart_png(hist_csv: str, out_png="price_chart.png", last_n=12) -> str:
    df = pd.read_csv(hist_csv)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").tail(last_n)

    plt.figure()
    plt.plot(df["month"], df["price_mid"])
    plt.title("Bangkok Skipjack Proxy (Last 12M)")
    plt.xlabel("Month")
    plt.ylabel("US$/MT")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


def send_email_with_inline_chart(subject: str, html_body: str, chart_path: str):
    if not (EMAIL_ADDR and EMAIL_PASSWORD and RECEIVER_ADDR):
        raise RuntimeError("EMAIL_ADDR / EMAIL_PASSWORD / RECEIVER_ADDR 설정 필요")

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDR
    msg["To"] = RECEIVER_ADDR

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html", _charset="utf-8"))
    msg.attach(alt)

    with open(chart_path, "rb") as f:
        img = MIMEImage(f.read(), _subtype="png")
    img.add_header("Content-ID", "<price_chart>")
    img.add_header("Content-Disposition", "inline", filename=os.path.basename(chart_path))
    msg.attach(img)

    import smtplib
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as server:
        server.login(EMAIL_ADDR, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDR, [RECEIVER_ADDR], msg.as_string())


def make_html(m0, price_info, pdf_url, catch_score, hist, pred) -> str:
    low, high = price_info["low"], price_info["high"]
    mid = round((low + high)/2, 2)

    delta_txt = ""
    if len(hist) >= 2:
        prev = hist.sort_values("month").iloc[-2]
        d = round(mid - float(prev["price_mid"]), 2)
        delta_txt = f"{'+' if d>=0 else ''}{d} (vs {pd.to_datetime(prev['month']).strftime('%Y-%m')})"

    pred_rows = ""
    if pred is None or pred.empty:
        pred_rows = "<tr><td colspan='4'>데이터 부족(권장: 12개월 이상)</td></tr>"
    else:
        for _, r in pred.iterrows():
            pred_rows += f"<tr><td>{r['month']}</td><td>{r['forecast']}</td><td>{r['low_95']}</td><td>{r['high_95']}</td></tr>"

    return f"""
    <html><body style="font-family: Arial, sans-serif;">
      <h2>🐟 Bangkok Skipjack Proxy + 3개월 예측(SARIMAX)</h2>
      <p><b>집계월:</b> {m0.strftime('%Y-%m')}</p>

      <h3>1) 최신 Proxy 시세 (US$/MT)</h3>
      <ul>
        <li><b>Low-High:</b> {low} ~ {high}</li>
        <li><b>Mid:</b> {mid} {f"/ <b>변동:</b> {delta_txt}" if delta_txt else ""}</li>
        <li><b>Catch score:</b> {round(float(catch_score),3)} (ITN 텍스트 기반)</li>
        <li><b>PDF:</b> <a href="{pdf_url}">{pdf_url}</a></li>
      </ul>

      <details>
        <summary>추출 근거 문장(가격)</summary>
        <div style="margin-top:8px; padding:10px; border:1px solid #ddd;">
          {price_info.get("sentence","")}
        </div>
      </details>

      <h3>2) 3개월 예측 (US$/MT)</h3>
      <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Month</th><th>Forecast</th><th>Low(95%)</th><th>High(95%)</th></tr>
        {pred_rows}
      </table>

      <h3>📈 최근 12개월 추세</h3>
      <img src="cid:price_chart" style="max-width:900px;width:100%;">

      <p style="margin-top:14px; color:#555; font-size:12px;">
        * 가격은 유료 방콕 CFR 벤치마크가 아니라 INFOFISH ITN의 Thailand delivery price 기반 Proxy입니다.<br/>
        * FAD는 룰 기반(7~8월=1) 자동 캘린더입니다.<br/>
        * Brent(월평균)은 FRED에서 자동 수집합니다.
      </p>
    </body></html>
    """


def main():
    m0 = month_start_kst()

    pdf_url = resolve_itn_pdf_url()
    pdf_bytes = download_bytes(pdf_url)
    text = extract_text_first_pages(pdf_bytes, max_pages=15)

    price_info = extract_skipjack_thailand_price(text)
    catch_score = calc_catch_score(text)

    hist = load_history(HISTORY_CSV)
    hist = upsert_history(hist, m0, price_info["low"], price_info["high"], catch_score, pdf_url)
    save_history(hist, HISTORY_CSV)

    train = build_train(hist)
    pred = sarimax_forecast_3m(train) if len(train) >= 6 else pd.DataFrame()

    chart_path = make_chart_png(HISTORY_CSV, "price_chart.png", last_n=12)

    subject = f"[참치 원어] Bangkok Proxy + 3개월 예측 ({m0.strftime('%Y-%m')})"
    html = make_html(m0, price_info, pdf_url, catch_score, hist, pred)
    send_email_with_inline_chart(subject, html, chart_path)

    print("OK:", subject)
    print("PDF:", pdf_url)
    print("PRICE:", price_info, "CATCH_SCORE:", catch_score)


if __name__ == "__main__":
    main()
