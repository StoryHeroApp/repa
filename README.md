# 🏠 REPA - Real Estate Personalized Assistant

A minimal AI-powered web app for matching apartment listings with user criteria. Built as a demo for the LangFlow-based Real Estate Personalized Assistant workflow.

## Features

- 💬 **Chat Interface** - Simple, clean UI for natural conversation
- 🔍 **Smart Criteria Extraction** - AI extracts structured data from natural language
- 🕷️ **Web Scraping** - Automatically fetches listing details from URLs
- 🖼️ **Image Analysis** - Analyzes apartment photos using vision AI
- ✅ **Match Scoring** - Generates detailed match reports with recommendations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and add:
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **Firecrawl API Key**: Get from [firecrawl.dev](https://firecrawl.dev)

### 3. Run the App

```bash
python app.py
```

The app will start at [http://localhost:8000](http://localhost:8000)

## Usage

1. Open [http://localhost:8000](http://localhost:8000) in your browser
2. Describe what you're looking for in an apartment
3. Include a listing URL from Homegate.ch or similar sites
4. Send your message and wait for the AI analysis!

### Example Input

```
I'm visiting Switzerland for ski season and want to hit the slopes! 
I got my wife and her family coming along, with our two sons. 
So it will be 6-7 people. We want to be right close to the ski action!

Check this listing: https://www.homegate.ch/rent/4002583790
```

### What You'll Get

- **Match Score** - Percentage match with your criteria
- **Detailed Analysis** - What matches and what doesn't
- **Property Highlights** - Key features of the listing
- **Image Analysis** - AI-powered photo descriptions
- **Recommendation** - Should you pursue this listing?
- **Next Steps** - Actionable advice

## How It Works

The app replicates your LangFlow workflow:

1. **Parse Input** - Extracts user criteria and listing URL
2. **Extract Criteria** - Uses GPT-4o-mini to structure requirements into JSON
3. **Scrape Listing** - Firecrawl fetches the full listing content
4. **Analyze Images** - GPT-4o-mini Vision analyzes apartment photos
5. **Generate Report** - GPT-4o-mini creates a comprehensive match analysis

## Project Structure

```
repa/
├── app.py                      # FastAPI backend server
├── static/
│   └── index.html             # Frontend chat UI
├── requirements.txt           # Python dependencies
├── .env.example              # Template for API keys
├── .env                      # Your API keys (create this)
├── REPA Iteration 1 v3.json  # Original LangFlow workflow
└── README.md                 # This file
```

## API Endpoints

- `GET /` - Serves the chat interface
- `POST /api/chat` - Processes chat messages
  - Request: `{ "message": "your message with criteria and URL" }`
  - Response: `{ "response": "AI analysis", "status": "success" }`

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **AI**: OpenAI GPT-4o-mini (text) + GPT-4o-mini (vision)
- **Web Scraping**: Firecrawl API
- **Environment**: python-dotenv

## Cost Considerations

- OpenAI API costs depend on usage (gpt-4o-mini is very affordable)
- Firecrawl offers a free tier for testing
- Image analysis is limited to 5 images per request for cost control

## Customization

You can adjust settings in `app.py`:

- `max_images` - Change number of images to analyze (default: 3)
- `model` - Switch between `gpt-4o` and `gpt-4o-mini`
- `temperature` - Adjust AI creativity (default: 0.1 for consistency)

## Development Notes

This is a **minimal demo** for course presentation. For production:

- Add authentication and rate limiting
- Implement caching for scraped listings
- Add database for storing searches
- Improve error handling and validation
- Add tests

## License

MIT - This is a student project for educational purposes.

## Credits

Built for Group Project Course - Real Estate Personalized Assistant
Based on LangFlow workflow architecture
