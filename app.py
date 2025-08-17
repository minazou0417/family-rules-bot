import os
import json
import time
import uuid
from typing import List, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
import re, unicodedata

# ====== オプション: Supabaseに保存したい人向け（無ければ自動でローカルJSONにフォールバック） ======
SUPABASE_AVAILABLE = False
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

# ====== EmbeddingはOpenAIを使用（text-embedding-3-small） ======
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

MODEL_EMBED = "text-embedding-3-small"
EMBED_DIM = 1536  # 現行の次元数

# ============= ユーティリティ ============= #

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# 文字正規化（表記ゆれ対策）
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============= データ層（Supabase or JSON） ============= #
class RuleStore:
    """家庭ルールのCRUDと、未登録質問ログの保存を抽象化"""

    def list_rules(self) -> List[Dict]:
        raise NotImplementedError

    def add_rule(self, title: str, content: str) -> Dict:
        raise NotImplementedError

    def delete_rule(self, rule_id: str) -> None:
        raise NotImplementedError

    def log_unknown(self, question: str) -> None:
        raise NotImplementedError


class JsonRuleStore(RuleStore):
    def __init__(self, rule_path: str = "rules.json", unknown_path: str = "unknowns.json"):
        self.rule_path = rule_path
        self.unknown_path = unknown_path
        # 初期ファイルが無い場合はサンプルから生成
        if not os.path.exists(self.rule_path):
            sample_path = "rules.sample.json"
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            with open(self.rule_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        if not os.path.exists(self.unknown_path):
            with open(self.unknown_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _read_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_rules(self) -> List[Dict]:
        return self._read_json(self.rule_path)

    def add_rule(self, title: str, content: str) -> Dict:
        rules = self.list_rules()
        rule = {"id": str(uuid.uuid4()), "title": title, "content": content}
        rules.append(rule)
        self._write_json(self.rule_path, rules)
        return rule

    def delete_rule(self, rule_id: str) -> None:
        rules = [r for r in self.list_rules() if r.get("id") != rule_id]
        self._write_json(self.rule_path, rules)

    def log_unknown(self, question: str) -> None:
        items = self._read_json(self.unknown_path)
        items.append({"id": str(uuid.uuid4()), "question": question, "ts": int(time.time())})
        self._write_json(self.unknown_path, items)


class SupabaseRuleStore(RuleStore):
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        # 以下のテーブルを事前に作成:
        #   rules(id uuid primary key, title text, content text)
        #   unknowns(id uuid primary key, question text, ts bigint)

    def list_rules(self) -> List[Dict]:
        res = self.client.table("rules").select("id,title,content").execute()
        return res.data or []

    def add_rule(self, title: str, content: str) -> Dict:
        rule = {"id": str(uuid.uuid4()), "title": title, "content": content}
        self.client.table("rules").insert(rule).execute()
        return rule

    def delete_rule(self, rule_id: str) -> None:
        self.client.table("rules").delete().eq("id", rule_id).execute()

    def log_unknown(self, question: str) -> None:
        item = {"id": str(uuid.uuid4()), "question": question, "ts": int(time.time())}
        self.client.table("unknowns").insert(item).execute()


# ============= Embedding層 ============= #
class Embedder:
    def __init__(self, api_key: Optional[str]):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openaiパッケージが見つかりません。requirements.txtを確認してください。")
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        resp = self.client.embeddings.create(model=MODEL_EMBED, input=texts)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vecs)


# ============= 検索 + 応答ロジック ============= #
class RuleAssistant:
    def __init__(self, store: RuleStore, embedder: Embedder, threshold: float = 0.62):
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self._cache_rules: List[Dict] = []
        self._cache_vecs: Optional[np.ndarray] = None
        self._owners: List[str] = []          # 各ベクトルがどのruleに属するか
        self._rules_by_id: Dict[str, Dict] = {}  # rule_id -> rule

    def _ensure_cache(self):
        rules = self.store.list_rules()
        if rules != self._cache_rules:
            self._cache_rules = rules
            self._rules_by_id = {r["id"]: r for r in rules}
            texts: List[str] = []
            owners: List[str] = []
            for r in rules:
                title = r.get("title", "")
                content = r.get("content", "")
                # タイトルの言い換えを複数パターンで埋め込み（言い回しに強くする）
                variants = [
                    f"{title}\n{content}",
                    f"{title}はいつ？",
                    f"{title}を教えて",
                    f"{title}とは？",
                ]
                texts.extend([normalize(v) for v in variants])
                owners.extend([r["id"]] * len(variants))
            self._cache_vecs = self.embedder.embed(texts) if texts else np.zeros((0, EMBED_DIM), dtype=np.float32)
            self._owners = owners

    def answer(self, question: str) -> Tuple[str, Optional[Dict], Optional[float]]:
        self._ensure_cache()
        if self._cache_vecs is None or self._cache_vecs.shape[0] == 0:
            self.store.log_unknown(question)
            return "まだルールが登録されていないみたい。教えてくれる？", None, None

        q_vec = self.embedder.embed([normalize(question)])[0]
        sims = np.array([cosine_sim(q_vec, v) for v in self._cache_vecs])
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        rule = self._rules_by_id.get(self._owners[idx])

        if best >= self.threshold:
            return f"『{rule['title']}』\n{rule['content']}", rule, best
        elif best >= 0.50:
            # ほどほどに近い時はサジェストを返す（任意ロジック）
            return f"もしかして『{rule['title']}』かな？\n{rule['content']}", rule, best
        else:
            self.store.log_unknown(question)
            return "そのルールはまだ知らないよ。教えてくれる？（あとで保護者の方が追加してね）", None, best

# ============= Streamlit UI ============= #
load_dotenv()

st.set_page_config(page_title="Family Rules Bot", page_icon="👨\u200d👩\u200d👧\u200d👦", layout="centered")
st.title("👨\u200d👩\u200d👧\u200d👦 家庭ルールボット")
st.caption("家庭のルールをAIがやさしくお知らせします")

# --- サイドバー（設定＆管理） ---
with st.sidebar:
    st.header("設定 / 管理")
    admin_pin = st.text_input("管理用PIN（任意）", type="password")
    st.write(":information_source: PINは簡易なUI切替のためのものです。厳格な認証が必要ならAuth導入を検討してください。")

# --- データストア選択（自動） ---
from pathlib import Path

def _get_secret(name: str, default=None):
    """st.secrets は secrets.toml が無いローカルで触るとエラーになるため、
    ファイルの存在を確認してから安全に参照するヘルパー。"""
    try:
        user_secrets = Path.home() / ".streamlit" / "secrets.toml"
        proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
        if user_secrets.exists() or proj_secrets.exists():
            return st.secrets.get(name, default)
    except Exception:
        pass
    return default

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or _get_secret("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")   or _get_secret("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")   or _get_secret("SUPABASE_KEY")
# 管理PINも環境変数/Secretsから取得（未設定なら 'admin'）
EXPECTED_PIN   = os.getenv("ADMIN_PIN")       or _get_secret("ADMIN_PIN", "admin")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が環境変数に設定されていません。Streamlit CloudのSecretsかローカルの.envに設定してください。")
    st.stop()

if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    store: RuleStore = SupabaseRuleStore(SUPABASE_URL, SUPABASE_KEY)
    st.sidebar.success("データ保存先: Supabase")
else:
    store = JsonRuleStore()
    st.sidebar.warning("データ保存先: ローカルJSON（デプロイ環境では揮発の可能性あり。DB導入推奨）")
    # デバッグ補助: ローカルJSONの実体パス
    st.sidebar.caption("rules.json → " + os.path.abspath("rules.json"))
    st.sidebar.caption("unknowns.json → " + os.path.abspath("unknowns.json"))

embedder = Embedder(api_key=OPENAI_API_KEY)
assistant = RuleAssistant(store, embedder, threshold=0.62)

# --- 管理UI（PINが "admin" など特定文字列で開く例） ---
if admin_pin == EXPECTED_PIN:
    st.subheader("🛠️ ルール管理（管理者向け）")
    with st.form("add_rule"):
        title = st.text_input("ルール名", placeholder="おやつの時間")
        content = st.text_area("ルール内容", placeholder="おやつは午後3時まで")
        submitted = st.form_submit_button("追加")
    if submitted and title.strip() and content.strip():
        store.add_rule(title.strip(), content.strip())
        st.success("ルールを追加しました！")

    # 既存ルール一覧
    rules = store.list_rules()
    if rules:
        st.write("### 登録済みルール")
        for r in rules:
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{r['title']}**\n\n{r['content']}")
            with cols[1]:
                if st.button("削除", key=f"del_{r['id']}"):
                    store.delete_rule(r['id'])
                    st.experimental_rerun()
    else:
        st.info("まだルールがありません。上のフォームから追加してください。")

    st.divider()
    st.write("### 未登録の質問（あとでルール化しましょう）")
# JsonRuleStore の場合はローカル unknowns.json を表示
if isinstance(store, JsonRuleStore) and os.path.exists("unknowns.json"):
    with open("unknowns.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    if items:
        for it in reversed(items[-50:]):
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(it.get("ts", 0)))
            st.write(f"- {ts} : {it['question']}")
    else:
        st.caption("記録なし")
# Supabase利用時はテーブルから取得して表示
elif isinstance(store, SupabaseRuleStore):
    try:
        res = store.client.table("unknowns").select("question,ts").order("ts", desc=True).limit(50).execute()
        rows = res.data or []
        if rows:
            for it in rows:
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(int(it.get("ts", 0))))
                st.write(f"- {ts} : {it['question']}")
        else:
            st.caption("記録なし（質問するとここに溜まります）")
    except Exception as e:
        st.warning(f"unknowns 取得に失敗しました: {e}")

# Supabaseモードで rules が空なら、サンプル投入ボタンを出す
if isinstance(store, SupabaseRuleStore):
    try:
        if not store.list_rules():
            if st.button("📥 サンプルルールをSupabaseへ投入"):
                try:
                    with open("rules.sample.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # 期待スキーマ: id/title/content
                    store.client.table("rules").insert(data).execute()
                    st.success("サンプルを投入しました。画面を更新して確認してください。")
                except Exception as e:
                    st.error(f"投入に失敗しました: {e}")
    except Exception:
        pass

st.divider()

# --- 子ども向けチャットUI ---
st.subheader("💬 きいてみよう！")
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("例: おやつ食べていい？ / テレビは何時まで？")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            try:
                answer, rule, sim = assistant.answer(prompt)
            except Exception as e:
                answer = f"エラーが発生しました: {e}"
                rule, sim = None, None
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})