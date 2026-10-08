# ClaimGuard AI — Insurance Claims Assessment & Decision Agent

An agentic insurance assistant built with **LangGraph + MCP + Groq/Gemini**. It answers policy and claim
questions, performs preliminary claim assessment (eligibility → payout → risk), and routes high-risk or
high-value decisions to a **human reviewer**. Data comes from **CSV files** with synthetic data.

```
User ──► FastAPI /webhook  or  cli.py
            │
            ▼
   LangGraph state machine  (SQLite checkpoints, Langfuse traces)
   START → [summarize?] → reasoning ──tool calls──► tools ──► [human_review?] ─┐
                              ▲                    │ pre-hook → MCP tools/call → post-hook
                              └────────────────────┴──────────────────────────────┘
                                                   │ stdio (JSON-RPC)
                                                   ▼
                                     MCP server (mcp_server/server.py)
                                                   │
                                     insurance_tools → data/*.csv
```

---

## 1. Where each task lives

| Task | What it does | File(s) |
|---|---|---|
| **1. State** | `AgentState` TypedDict with reducers: messages, summary, tool log, risk records, pending review, human decisions | `claimguard/state.py` |
| **2. Reasoning node** | LLM bound to tools + system prompt; outputs tool calls (logged as JSON-RPC `tools/call`) or final text; never executes | `claimguard/reasoning.py`, `claimguard/prompts.py` |
| **3. Tools (JSON-RPC)** | 9 Python functions → JSON schemas; tools node routes each call to the function and appends a `ToolMessage` | `insurance_tools/tools.py`, `insurance_tools/rules.py`, `claimguard/tool_executor.py`, `LocalToolProvider` in `claimguard/tool_provider.py` |
| **4. MCP** | Standard MCP server; agent's MCP client sends `tools/list` on startup and `tools/call` over stdio (or HTTP) | `mcp_server/server.py`, `MCPToolProvider` in `claimguard/tool_provider.py` |
| **5. Hooks** | Pre-hook: schema, ID formats, amounts, authorization → simulated **400/403**. Post-hook: business invariants (**422**), PII masking, HITL trigger | `claimguard/hooks.py` |
| **6. HITL** | Conditional edge → `human_review` node → `interrupt()` pauses; resume with Approve/Reject from terminal or API | `claimguard/hitl.py`, `cli.py`, `POST /webhook/review` |
| **7. Compaction + checkpoints** | If `len(messages) > 10`, a separate LLM call summarizes the first 8 and deletes them; `AsyncSqliteSaver` persists every step to disk | `claimguard/compaction.py`, `claimguard/agent.py` |
| **8. Telemetry** | Langfuse v3: root trace per run, LLM generations (tokens, latency), one `tool:<name>` observation per call with the JSON-RPC request/response | `claimguard/telemetry.py` |
| **9. Eval + Deploy** | 10 scenarios → traces → LLM-as-a-judge grades Error Recovery 1–5; FastAPI + Dockerfile for Render / HF Spaces | `eval/`, `api/main.py`, `Dockerfile` |

```
claimguard-ai/
├── data/                  policies.csv, claims.csv, claim_history.csv  (dummy data)
├── insurance_tools/       CSV access, business rules, tool functions (the "external system")
├── mcp_server/server.py   MCP server exposing the tools
├── claimguard/            the agent (state, reasoning, tools node, hooks, HITL, compaction, telemetry, graph, runner)
├── api/main.py            FastAPI app
├── cli.py                 terminal chat with Approve/Reject prompt
├── eval/                  scenarios.json, run_scenarios.py, llm_judge.py
├── tests/                 offline tests (no API key needed)
├── Dockerfile, requirements.txt, .env.example
```

---

## 2. The CSV data source

| File | Columns |
|---|---|
| `policies.csv` | policy_id, customer_name, customer_phone, customer_email, policy_type, premium_annual, coverage_amount, deductible, policy_status, start_date, end_date, covered_perils (`|`-separated) |
| `claims.csv` | claim_id, policy_id, claim_type, claim_amount, incident_date, reported_date, claim_status, description, decision_note, updated_at |
| `claim_history.csv` | history_id, policy_id, claim_date, claim_type, claim_amount, claim_status |

Dates are `YYYY-MM-DD`. You can edit the CSVs while the agent runs; every tool call re-reads them.
`update_claim_status` writes back to `claims.csv` (keep a copy of the original if you want to reset the demo).

**Built-in demo cases**

| Claim | Situation | Expected outcome |
|---|---|---|
| C1001 | Small auto accident, clean history | LOW risk, fast-track |
| C1003 | ₹4.5L theft, 2 previous claims (₹5.2L paid) | **HIGH risk (80) → human review** |
| C1004 | ₹12L flood, low risk | **High value → human review** |
| C1005 / C1006 | Expired / lapsed policy | Not eligible |
| C1007 | ₹35L fire 16 days after policy start | MEDIUM risk + high value → review |
| C1009 | Theft on a policy that covers only Accident/Fire | Not eligible (peril not covered) |
| C1011 | Incident before policy start | Not eligible |
| C1012 | ₹9.5L claim on ₹8L coverage | Payout capped at ₹7,90,000; HIGH risk → review |
| C1013 | Reported 50 days late | Late-reporting risk factor |

Risk scoring (in `insurance_tools/rules.py`): ≥2 previous claims +30, 1 previous +10, previous payouts >50% of
coverage +20, claim >50% of coverage +20, claim ≥ ₹5L +15, incident within 30 days of start +20, theft +10,
reported >30 days late +10. LOW < 30 ≤ MEDIUM < 60 ≤ HIGH. Payout = min(claim, coverage) − deductible.

---

## 3. Setup on your PC

**Prerequisites:** Python 3.11 or 3.12, Git, (optional) Docker Desktop.

```bash
cd claimguard-ai
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # Windows: copy .env.example .env
```

Get the free keys and put them in `.env`:

1. **Groq** — https://console.groq.com/keys → `GROQ_API_KEY`. (Or Gemini at https://aistudio.google.com/apikey → set `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY`.)
2. **Langfuse** — sign up at https://cloud.langfuse.com → create a project → *Settings → API Keys* → copy public/secret keys and the host shown there (EU `https://cloud.langfuse.com` or US `https://us.cloud.langfuse.com`).

Verify the build (no keys needed):

```bash
pytest -q
```

You should see `8 passed`. These tests run the real MCP server and prove the 400 self-correction,
the HITL pause surviving a restart, and compaction, using a scripted fake LLM.

---

## 4. Run it and verify each task

```bash
python cli.py
```

The banner shows `tools via MCP: 9 discovered` — that count came from the MCP `tools/list` request (Task 4).
Use `/tools` to list them, `/state` to see the checkpointed state, `/new` for a new thread.

Try these in order:

| Type this | What it demonstrates |
|---|---|
| `What is the premium for P1003?` | Level 1 info retrieval: one tool, short answer (Tasks 2–4) |
| `When does policy P1003 expire?` | Same tool, answers only what was asked |
| `Tell me everything about policy P1003` | Full record; note phone/email are **masked** by the post-hook (Task 5) |
| `What's the status of claim c1003?` | If the model sends `c1003`, the pre-hook returns **400** and the model retries with `C1003` — the CLI prints `errors self-corrected: 1` (Task 5) |
| `Is claim C1009 eligible?` | Eligibility with failed check (theft not covered) |
| `What would the payout for C1001 be if the amount were -50000?` | Pre-hook rejects the negative amount (400); the agent explains |
| `Assess claim C1003 and tell me whether it should proceed` | Multi-tool assessment → HIGH risk → **graph pauses** and shows the review card → type `Approve` or `Reject` → agent records the decision in `claims.csv` and summarises (Task 6). The run is long enough that **context compaction** usually triggers (`context compacted` in the CLI line) (Task 7) |
| `Approve claim C1001` | The pre-hook first returns 400 *PrerequisiteMissing* ("assess first"); the agent assesses, sees LOW risk, and is allowed to approve it itself |
| `Approve claim C1003 right now` | Guardrails stop the agent from deciding alone: it must assess first (400), the assessment flags HIGH risk, and the graph pauses for a human. If the model ever tries to approve without a human decision, the pre-hook returns **403** (defence in depth) |

**Checkpoint survival (Task 7):** start an assessment of `C1004`, and when the approval prompt appears press
`Ctrl+C`. Then run `python cli.py --thread <the-thread-id-shown>` — the agent reloads the paused state from
`data/checkpoints.sqlite` and asks for the decision again.

**Langfuse (Task 8):** open your Langfuse project → *Tracing*. Each request is a `claimguard-agent` trace
(session = thread id). Inside you'll see each LLM generation with token counts and latency, and a
`tool:<name>` observation whose **input is the JSON-RPC `tools/call` request** and **output is the JSON-RPC
response**; failed calls are marked level ERROR with the 400/403/404 message.

---

## 5. The API

```bash
uvicorn api.main:app --reload --port 7860
```

Open http://localhost:7860/docs for the interactive Swagger UI.

```bash
# Ask a question
curl -X POST localhost:7860/webhook -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","message":"Assess claim C1003 and tell me whether it should proceed"}'
# → {"status":"awaiting_human_review","review":{"claim_id":"C1003","risk_level":"HIGH",...}, ...}

# Human decision (resumes the paused graph)
curl -X POST localhost:7860/webhook/review -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","decision":"reject","comment":"Too many prior theft claims"}'

# Inspect persisted state
curl localhost:7860/threads/demo-1
```

Add `?debug=true` to `/webhook` to include every JSON-RPC event in the response. If you set `API_KEY` in
`.env`, send it as header `X-API-Key`.

**Optional — MCP server over HTTP:** in one terminal run
`MCP_TRANSPORT=streamable-http python -m mcp_server.server` (listens on `127.0.0.1:8001/mcp`), and in
`.env` set `MCP_SERVER_URL=http://127.0.0.1:8001/mcp`. The agent then discovers and calls tools over HTTP
instead of spawning the stdio server. You can also inspect the server with the official MCP Inspector:
`npx @modelcontextprotocol/inspector python -m mcp_server.server`.

---

## 6. Evaluation (Task 9.1)

```bash
python -m eval.run_scenarios     # runs 10 error-prone scenarios, auto-answers HITL, creates traces
python -m eval.llm_judge         # grades the 10 latest traces for Error Recovery (1–5)
```

The scenarios (`eval/scenarios.json`) deliberately include lowercase IDs, a missing `P` prefix, a
non-existent claim (404), a negative amount, an invalid ID `PX12`, a forbidden approval (403) and two HITL
cases. The judge reads traces **from Langfuse** when keys are configured (falls back to the local copy in
`data/traces/traces.jsonl`) and writes `eval/results/report.md` and `report.json` with per-trace scores,
reasoning and the average. Use `--source local` or `--source langfuse` to force a source.

Free-tier tip: Groq has per-minute token limits; the scripts pause between scenarios. If you hit a 429,
re-run after a minute.

---

## 7. Docker & deployment (Task 9.2)

**Local container**

```bash
docker build -t claimguard-ai .
docker run -p 7860:7860 --env-file .env claimguard-ai
# open http://localhost:7860/docs
```

**Render**

1. Push the project to a GitHub repo (`.env` is git-ignored — never commit keys).
2. Render dashboard → *New → Web Service* → connect the repo → Runtime **Docker**.
3. *Environment*: add `GROQ_API_KEY`, `LLM_PROVIDER`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
   `LANGFUSE_HOST`, and optionally `API_KEY`. Render sets `PORT` automatically; the Dockerfile uses it.
4. Deploy, then test: `curl https://<your-app>.onrender.com/health` and the `/webhook` calls above.

**Hugging Face Spaces**

1. Create a Space → SDK **Docker** → blank template.
2. The Space's `README.md` must start with this front matter (put it at the top of the README in the Space repo):
   ```yaml
   ---
   title: ClaimGuard AI
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push the project files to the Space repo (git or the web uploader).
4. *Settings → Variables and secrets*: add the same keys as **secrets**.
5. When the build finishes, open `https://<user>-<space>.hf.space/docs` and test `/webhook`.

Note: free hosts have ephemeral disks, so `claims.csv` edits and `checkpoints.sqlite` reset when the
container restarts or redeploys. Checkpoints do survive a normal process restart locally. For persistent
production storage, point `CHECKPOINT_DB`/`DATA_DIR` at a mounted disk (Render paid disks) or swap in a
Postgres checkpointer.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `tool_use_failed` / malformed tool call from Groq | The node retries once automatically. If frequent, try another tool-capable Groq model via `GROQ_MODEL`, or use Gemini. |
| 429 rate limit | Free tier limits; wait a minute, or lower usage by switching `GROQ_SUMMARY_MODEL` to a smaller model. |
| No traces in Langfuse | Check keys and `LANGFUSE_HOST` region (EU vs US). Traces are flushed after each request. |
| `ModuleNotFoundError: mcp.server.fastmcp` | You installed `mcp` 2.x. Run `pip install "mcp>=1.9,<2"` (pinned in requirements). |
| Thread returns 409 | That thread is paused for review. Send `POST /webhook/review` first, or use a new `thread_id`. |
| Reset demo data | Restore the original `data/claims.csv` (e.g. `git checkout data/claims.csv`) and delete `data/checkpoints.sqlite`. |
