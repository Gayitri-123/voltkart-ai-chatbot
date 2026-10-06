# VoltKart Customer Chatbot (Amazon Bedrock + OpenAI)

A customer support chatbot. Customers can:
- Search products and ask about specs, prices, offers and stock
- Track orders (order ID + registered phone number)
- Check refund/return status
- Ask about return, refund, cancellation, delivery, warranty and payment policies

The customer picks the AI model in the sidebar and can switch models mid-conversation:
- Claude on Amazon Bedrock: Claude Opus 5, Sonnet 5, Haiku 4.5, Opus 4.8, Opus 4.7
- More models on Amazon Bedrock (Converse API): Amazon Nova Pro, Nova 2 Lite, Llama 4 Maverick, Mistral Large 3,
  DeepSeek V3.2, Qwen3 Next 80B, GPT-OSS 120B, GPT-6 Astra, Grok 4.6
- OpenAI: GPT-5, GPT-5 mini, GPT-4.1, GPT-4o, GPT-4o mini (enabled when `OPENAI_API_KEY` is set)

## How it works
- `app.py`: FastAPI server (`/api/models`, `/api/chat`, `/api/reset`).
- `providers.py`: model list, system prompt, and the tool-calling loop for each provider: Claude on Bedrock
  (Anthropic SDK), other Bedrock models (boto3 Converse API) and OpenAI.
  Tool calls are saved in the chat history, so a newly selected model sees what the previous model looked up.
- `tools.py`: the tools Claude can call (`search_products`, `get_product_details`, `get_order_status`, `get_policy`).
- `data.py`: **sample** catalog, orders and policies. Replace with real APIs/DB.
- `static/`: chat web UI (`index.html`, `styles.css`, `app.js`). Responsive, light/dark mode.

## Run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8000
```
Open http://localhost:8000

Uses your AWS credentials (`~/.aws/credentials`, env vars, or IAM role). Optional env vars:
- `AWS_REGION` (default `us-east-1`)
- `STORE_NAME` (default `VoltKart`): store name shown in the UI and used by the assistant
- `OPENAI_API_KEY`: enables the OpenAI models (can also be put in a `.env` file in the project folder, which is gitignored)

To add or remove models, edit `MODELS` in `providers.py`. Any Bedrock model that supports tool use in the
Converse API can be added with `"provider": "bedrock_converse"`; it must be enabled for your AWS account and region.

## Comparing models (evals)
`evals/run_eval.py` runs the test conversations in `evals/cases.json` against each model, using the same
tool-calling loop as the app, and prints a comparison table (pass rate overall and per category, LLM judge score,
latency, tool calls, errors).
```bash
python evals/run_eval.py                                            # all available models, all cases
python evals/run_eval.py --models claude-sonnet-5 gpt-5 nova-pro --judge claude-opus-5
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
chatbot's system prompt and the case's `notes`. Use a strong model (`claude-opus-5`); smaller models judge
inconsistently. Full transcripts, checks and scores are saved to `evals/results/<timestamp>/` (gitignored).
Every model × case is a real API call, so running all models costs money; start with `--models`.

## Test order data
| Order | Phone | Status |
|---|---|---|
| ORD10001 | 9876543210 | Delivered |
| ORD10002 | 9876543210 | Out for delivery |
| ORD10003 | 9123456780 | Return requested, refund pending pickup |

## Going to production
- Connect `tools.py` to the real product catalog, order management system and refund service.
- Store chat sessions in Redis/DynamoDB instead of memory.
- Add proper customer login (e.g. OTP) instead of only checking the phone number.
