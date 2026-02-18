# Configuration Secrets

Store local API keys or tokens here so they remain outside of version control. The `.gitignore` entry excludes any `*.env`, `*.key`, or `*.token` files inside this directory.

## Usage

1. Copy `hf.env.example` to `hf.env` (or another filename ending with `.env`).
2. Edit the new file and paste your Hugging Face token.
3. Load the values before running the backend, e.g.:
   - `setx HUGGINGFACEHUB_API_TOKEN <your token>` (Windows) or
   - `export HUGGINGFACEHUB_API_TOKEN=$(grep HUGGINGFACEHUB_API_TOKEN backend/app/config/hf.env | cut -d'=' -f2-)` (macOS/Linux).
4. The FastAPI service can then access the token via the standard `HUGGINGFACEHUB_API_TOKEN` environment variable.

> Never commit the real token files—only the template should stay under version control.
