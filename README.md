# VoltKart Customer Chatbot (Gemini, open-weights and Claude)

A customer support chatbot. Customers can:
- Search products and ask about specs, prices, offers and stock
- Track orders (order ID + registered phone number)
- Check refund/return status
- Ask about return, refund, cancellation, delivery, warranty and payment policies

Three sources of models are offered (only models that support tool calling, needed to look up products and orders):

| Source | Models | Credentials |
|---|---|---|
| [Gemini API](https://ai.google.dev/gemini-api/docs/pricing) (free tier) | Gemini 3.8 Flash (default), 3.7 Flash, 3.6 Flash, 3.5 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 3 Flash Preview, 2.5 Flash | `GEMINI_API_KEY` from https://aistudio.google.com/apikey |
| [Ollama](https://ollama.com) (free, runs on your machine) | Open-weights models you have pulled that support tools, e.g. Qwen3, Llama 3.1, Mistral, gpt-oss | None: start Ollama and `ollama pull <model>` |
| [Amazon Bedrock](https://aws.amazon.com/bedrock/pricing/) (**paid**, billed to your AWS account) | Claude Fable 5.1, Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5 | AWS credentials (env vars or `~/.aws`) with Bedrock model access |

The customer picks the model in the sidebar and can switch models mid-conversation. Models whose credentials aren't
set show as "Not configured", so you can enable only the source you want. Ollama models are found at runtime: every
pulled model that supports tool calling is listed (refresh the page after pulling one); none are listed if Ollama
isn't running.

The Gemini free tier has limits (requests per minute/day). When a model hits its limit, the customer is asked to pick
another model. Free-tier prompts may be used by providers to improve
their products (Google does), so don't send real customer data through them.

## How it works
- `app.py`: FastAPI server (`/api/models`, `/api/chat`, `/api/reset`). Each `/api/chat` reply has a `route`: the kind
  of lookup the reply started with (`product`, `order`, `policy`, or `chat` when nothing was looked up yet), used for
  gray-box testing. For white-box testing, `"debug": true` with an `X-Debug-Token` header equal to the
  `DEBUG_TOKEN` env var also returns the tool calls behind the reply, with phone numbers and emails masked. Without
  `DEBUG_TOKEN` set (the default, and what production should use) no caller gets traces.
- Conversations belong to the browser that started them: the server gives each browser a random owner id in an
  HttpOnly cookie, and answers 404 when a session id is used or reset without its owner's cookie (so a leaked or
  guessed session id is useless). A message sent while the previous reply in the same conversation is still being
  generated gets 409; other conversations are not held up.
- `providers.py`: model list, system prompt, and the tool-calling loop: Gemini and Ollama through their
  OpenAI-compatible APIs,
  Claude through the Anthropic SDK on Bedrock.
  Tool calls are saved in the chat history, so a newly selected model sees what the previous model looked up.
- `tools.py`: the tools the model can call (`search_products`, `get_product_details`, `get_order_status`, `get_policy`).
- `data.py`: **sample** catalog, orders and policies. Replace with real APIs/DB.
- `static/`: chat web UI (`index.html`, `styles.css`, `app.js`). Responsive, light/dark mode.

## Run
Put a free Gemini key in a `.env` file in the project folder:
```bash
GEMINI_API_KEY=...
```
For the Claude models, configure AWS credentials (`aws configure`, or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in
`.env`) for an account with Bedrock access to the Claude models. Claude usage is billed to that AWS account.

For open-weights models, install [Ollama](https://ollama.com/download) and pull a model that supports tools:
```bash
ollama pull qwen3        # or llama3.1, mistral, qwen2.5, gpt-oss:20b ...
```
Local models run on your CPU/GPU, so replies can take from a few seconds to minutes depending on the hardware and
model size (the first reply also waits for the model to load). Ollama only works when the chatbot runs on the same
machine (or `OLLAMA_URL` points to a reachable Ollama server); the Render free plan can't run it. No customer data
leaves the machine.

Then start the chatbot:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8000
```
Open http://localhost:8000

Optional env vars (can also be put in a `.env` file in the project folder, which is gitignored):
- The credentials above: each enables that source's models
- `OLLAMA_URL` (default `http://localhost:11434`): Ollama server for the open-weights models
- `OLLAMA_TIMEOUT` (default `300`): seconds to wait for a local model's reply
- `AWS_REGION` (default `us-east-1`): Bedrock region for the Claude models
- `STORE_NAME` (default `VoltKart`): store name shown in the UI and used by the assistant
- `DEBUG_TOKEN` (default unset, so debug traces are off): operator secret for `"debug": true` tool traces

To add or remove models, edit `_MODEL_LIST` in `providers.py`: `(id, label, provider, model name in that provider's
API)`. Only add models that support tool calling. Ollama models need no entry: pull or remove them with `ollama`.

## Comparing models (evals)
`evals/run_eval.py` runs the test conversations in `evals/cases.json` against each model, using the same
tool-calling loop as the app, and prints a comparison table (pass rate overall and per category, LLM judge score,
latency, tool calls, errors).
```bash
python evals/run_eval.py                                            # all available models, all cases
python evals/run_eval.py --models gemini-3.8-flash claude-haiku-5-5 --judge claude-sonnet-5-5
python evals/run_eval.py --models ollama/qwen3:latest --workers 1  # a local model (ids as shown by /api/models)
python evals/run_eval.py --cases order_ policy_ --workers 8          # only cases whose id starts with these
```
A case passes when all of its automatic checks pass:
- `expect_tools` / `must_call_any` / `forbid_tools`: which tools must (or must not) be called
- `expect_args`: tool arguments, e.g. the right order ID
- `must_match` / `must_not_match`: regexes on the final reply / all replies (lowercased, digit commas removed, so
  "Rs 19,999" matches `19999`)
- Price grounding (on by default, `"check_prices": false` to turn off): every rupee amount in a reply must come from
  a tool result or the customer's message, or be a simple difference (savings) or EMI split of those figures

`--judge <model>` also has that model score each conversation 1-5 for correctness, helpfulness and tone, using the
chatbot's system prompt and the case's `notes`. Use a strong model such as GPT-OSS 120B; small models
judge inconsistently. With the free tier, keep `--workers` low (e.g. 2) to stay under the rate limits. Full transcripts, checks and scores are saved to `evals/results/<timestamp>/` (gitignored).

## Test order data
| Order | Phone | Status |
|---|---|---|
| ORD10001 | 9876543210 | Delivered |
| ORD10002 | 9876543210 | Out for delivery |
| ORD10003 | 9123456780 | Return requested, refund pending pickup |

## Going to production
- Connect `tools.py` to the real product catalog, order management system and refund service.
- Store chat sessions in Redis/DynamoDB instead of memory.
- Add proper customer login (e.g. OTP) instead of only checking the phone number. Then bind conversations to the logged-in
  customer instead of the anonymous owner cookie.
- Leave `DEBUG_TOKEN` unset, or set it to a long random secret known only to operators.
