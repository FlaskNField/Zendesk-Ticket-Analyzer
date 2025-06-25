# Zendesk-Ticket-Analyzer & Dashboard

This project provides a Python Tkinter dashboard for analyzing Zendesk support tickets using AI (OpenAI GPT-4, Kapa.ai) and visualizing results with interactive charts and drilldowns.

## Features
- Fetches Zendesk tickets for a configurable timeframe and form type
- Analyzes tickets with OpenAI for summaries, self-service potential, and sentiment
- Compares Kapa.ai chatbot answers to agent answers
- Assigns high-level themes to tickets
- Displays results in a modern, interactive Tkinter dashboard
- Drilldown to ticket details, with clickable ticket links
- Option to run from a previously generated CSV to avoid repeated API calls
<img width="1514" alt="image" src="https://github.com/user-attachments/assets/4b51a394-def2-4a99-a685-7e8dc375fe6a" />

## Setup
1. **Clone the repository**
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   # or manually:
   pip install pandas openai spacy tenacity python-dotenv matplotlib tqdm
   python -m spacy download en_core_web_md
   ```
3. **Configure your environment variables**
   - Create a `.env` file in the project root with the following format:
     ```ini
     ZENDESK_API_TOKEN=your_zendesk_api_token
     ZENDESK_EMAIL=your_email@example.com
     ZENDESK_SUBDOMAIN=your_subdomain
     OPENAI_API_KEY=your_openai_api_key
     KAPA_API_KEY=your_kapa_api_key
     KAPA_PROJECT_ID=your_kapa_project_id
     KAPA_INTEGRATION_ID=your_kapa_integration_id
     ```
   - **Never commit your real .env file or API keys to a public repo!**

## Usage
Run the script from your virtual environment:
```bash
python analyze_tickets.py
```

### Options
- `--ticket-type` (default: account-billing)
- `--date-range` (default: 24hr)
- `--csv` (use a previously generated CSV)
- `--no-gui` (skip launching the dashboard)

Example:
```bash
python analyze_tickets.py --ticket-type technical-support --date-range last-week
```

## Notes
- The dashboard requires a working internet connection and valid API credentials.
- All sensitive data is loaded from environment variables or a `.env` file.
- For best results, use a recent version of Python (3.8+ recommended).

## License
MIT License 
