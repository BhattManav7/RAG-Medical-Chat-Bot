# RAG-Medical-Chat-Bot

## Quick start

### Backend

1. Create a Hugging Face token file:
	- Copy [backend/app/config/hf.env.example](backend/app/config/hf.env.example) to `backend/app/config/hf.env`.
	- Replace `YOUR_HF_TOKEN_HERE` with your Hugging Face token.
2. Create and activate a Python virtual environment.
3. Install dependencies:
	- `pip install -r backend/requirements.txt`
4. Run the API:
	- `uvicorn app.main:app --reload --app-dir backend`

The API should be available at `http://localhost:8000`.

### Frontend

1. From `RAG-Medical-Chat-Bot/frontend`, install dependencies:
	- `npm install`
2. Start the dev server:
	- `npm run dev`

If your backend runs on a different URL, set `VITE_API_BASE_URL` in `frontend/.env`.
