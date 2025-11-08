#!/usr/bin/env python3
import argparse, io, sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LIKELY_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

def try_read_csv(path):
    for enc in LIKELY_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"Read CSV with encoding: {enc} (columns: {list(df.columns)[:6]})")
            return df, enc
        except Exception as e:
            print(f"Failed to read with encoding {enc}: {type(e).__name__}: {e}")
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    print("Read CSV by decoding bytes with utf-8 (errors=replace).")
    return df, "utf-8 (errors=replace)"

def normalize_df(df):
    cols = list(df.columns)
    lower = [str(c).lower() for c in cols]
    if 'user' in lower and 'text' in lower:
        return df
    if len(cols) >= 6 and ('user' not in lower or 'text' not in lower):
        print("Assuming headerless Sentiment140 format and reassigning columns.")
        newcols = ['sentiment','tweet_id','date','query','user','text'] + [f'col_{i}' for i in range(6, len(cols))]
        df.columns = newcols
        return df
    renamed = False
    for c in cols:
        cl = str(c).lower()
        if 'text' in cl or 'tweet' in cl:
            if 'text' not in lower:
                df = df.rename(columns={c: 'text'}); renamed = True
        if 'user' in cl or 'screen_name' in cl or 'username' in cl:
            if 'user' not in lower:
                df = df.rename(columns={c: 'user'}); renamed = True
    if not renamed and ('user' not in df.columns or 'text' not in df.columns):
        print("ERROR: couldn't find 'user' and 'text' columns after normalization. Columns were:", df.columns.tolist())
        sys.exit(1)
    return df

def compute_scores(df):
    if 'sentiment' in df.columns:
        try:
            uniq = set(int(float(x)) for x in df['sentiment'].dropna().unique()[:10])
        except Exception:
            uniq = set()
        if uniq and uniq.issubset({0,4}):
            df['score'] = df['sentiment'].astype(int).map({0:-1.0,4:1.0})
            print("Using existing sentiment labels (0/4) mapped to -1/1.")
            return df
        try:
            smin, smax = float(df['sentiment'].min()), float(df['sentiment'].max())
            if -1.0 <= smin <= 1.0 and -1.0 <= smax <= 1.0:
                df['score'] = df['sentiment'].astype(float)
                print("Using existing sentiment column as float scores in [-1,1].")
                return df
        except Exception:
            pass
    print("Computing sentiment with VADER (fallback).")
    analyzer = SentimentIntensityAnalyzer()
    df['text'] = df['text'].astype(str)
    df['score'] = df['text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])
    return df

def aggregate_per_user(df, out_path):
    df['user'] = df['user'].astype(str)
    per_user = df.groupby('user')['score'].mean().reset_index()
    per_user.columns = ['user_id','sentiment']
    per_user.to_csv(out_path, index=False)
    print(f"Wrote per-user sentiment to {out_path}, users: {len(per_user)}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="data/per_user_sentiment.csv")
    args = p.parse_args()
    print("Loading", args.input)
    df, enc = try_read_csv(args.input)
    df = normalize_df(df)
    df = compute_scores(df)
    aggregate_per_user(df, args.out)

if __name__ == "__main__":
    main()
