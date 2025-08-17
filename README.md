# Family Rules Bot (Streamlit)

家庭のルールをAIが答えるミニアプリ。Streamlitで実装し、OpenAIの埋め込みでルール検索します。

## ✅ 要件
- Python 3.10 以上
- OpenAI API Key（`text-embedding-3-small` を使用）
- （任意）Supabase プロジェクト

## 📦 セットアップ（ローカル）
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # .env を編集して OPENAI_API_KEY を設定
streamlit run app.py
```

## 🧪 使い方
- サイドバーに PIN として `admin` を入れると管理画面が開き、ルールを追加・削除できます。
- ルールが無い／一致しない質問は `unknowns.json`（または Supabase `unknowns` テーブル）に記録されます。
- 既定の初期データは `rules.sample.json` を `rules.json` にコピーして読み込みます。

## ☁️ デプロイ（Streamlit Community Cloud）
1. GitHub にリポジトリを作成し、本プロジェクトを push。
2. https://streamlit.io/cloud へログイン → **New app** → 対象リポジトリとブランチを選択。
3. **Advanced settings → Secrets** に以下を登録：
   ```
   OPENAI_API_KEY = sk-...（必須）
   SUPABASE_URL = https://xxxx.supabase.co（任意）
   SUPABASE_KEY = xxxxx（任意）
   ```
4. Deploy を押すと公開。エントリポイントは `app.py`。

> 注: Streamlit Cloud は一時ストレージのため、JSON保存はリスタートで消える場合があります。永続化したい場合は Supabase を設定してください。

## 🗄️ Supabase スキーマ（任意）
```sql
create table if not exists rules (
  id uuid primary key,
  title text not null,
  content text not null
);
create table if not exists unknowns (
  id uuid primary key,
  question text not null,
  ts bigint not null
);
```

## 🔐 .gitignore（推奨）
```
# Python
.venv/
__pycache__/
*.pyc

# Env & local data
.env
rules.json
unknowns.json

# Streamlit
.streamlit/
```

## 🧰 開発Tips
- 類似度しきい値は `RuleAssistant(threshold=0.70)` を調整。
- ルール文言は「短いタイトル＋本文」で登録すると検索しやすいです。
- 子供向けの口調にしたい場合は `answer` をテンプレで包む（例：「〜だよ！」）。

## 🚀 拡張アイデア
- **声でQ&A**: `streamlit-webrtc` + OpenAI STT/TTS。
- **未登録質問ダッシュボード**: `unknowns` を表表示＆「この内容でルール追加」ボタン。
- **キャラクター口調**: サイドバーで「標準/関西弁/ゆるい」など切替。
- **簡易認証**: `st.secrets` に PIN を置き、管理UI制御。
