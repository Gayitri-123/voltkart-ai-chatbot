# VoltKart Customer Chatbot (Amazon Bedrock + OpenAI GPT-4o)

A customer support chatbot. Customers can:
- Search products and ask about specs, prices, offers and stock
- Track orders (order ID + registered phone number)
- Check refund/return status
- Ask about return, refund, cancellation, delivery, warranty and payment policies

The customer picks the AI model in the sidebar and can switch models mid-conversation:
- Amazon Bedrock: Claude Opus 5, Claude Sonnet 5, Claude Haiku 4.5
- OpenAI: GPT-4o, GPT-4o mini (enabled when `OPENAI_API_KEY` is set)

## How it works
- `app.py`: FastAPI server (`/api/models`, `/api/chat`, `/api/reset`).
- `providers.py`: model list, system prompt, and the tool-calling loop for Bedrock and OpenAI.
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
- `OPENAI_API_KEY`: enables GPT-4o and GPT-4o mini

To add or remove models, edit `MODELS` in `providers.py`.

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
