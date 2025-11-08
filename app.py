"""
REPA - Real Estate Personalized Assistant
A minimal demo app for the LangFlow-based apartment matching workflow
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
import requests
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="REPA - Real Estate Personalized Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


def extract_url_from_message(message: str) -> tuple[str, str]:
    """Extract URL from message and return cleaned message and URL"""
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, message)
    
    if urls:
        # Get the first URL
        url = urls[0]
        # Remove URL from message
        clean_message = re.sub(url_pattern, '', message).strip()
        return clean_message, url
    
    return message, ""


def call_firecrawl_scraper(url: str) -> dict:
    """Scrape the listing URL using Firecrawl API"""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY not found in environment")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": url,
        "formats": ["markdown", "html"]
    }
    
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            data = result.get("data", {})
            return {
                "content": data.get("markdown", data.get("html", "")),
                "url": url,
                "metadata": data.get("metadata", {}),
                "title": data.get("metadata", {}).get("title", ""),
                "description": data.get("metadata", {}).get("description", ""),
            }
        else:
            return {"error": result.get("error", "Unknown error")}
    
    except Exception as e:
        return {"error": str(e)}


def extract_criteria_with_openai(user_message: str) -> dict:
    """Extract apartment criteria from user message using OpenAI"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    system_prompt = """You are an expert at extracting structured apartment rental criteria from natural language.

Extract information from the user's request and return it as valid JSON.

IMPORTANT: Only include fields that the user explicitly mentions. Do NOT include fields with null values.

Available field names you may use (only if mentioned):
- location: The city, postal code, area, or proximity requirement (string)
- min_rooms: Minimum number of rooms (number)
- max_rooms: Maximum number of rooms (number)
- min_living_space: Minimum living space in square meters (number)
- max_living_space: Maximum living space in square meters (number)
- min_rent: Minimum rent in CHF (number)
- max_rent: Maximum rent in CHF (number)
- occupants: Number of people who will live there (number)
- duration: How long they need it (string, e.g., "ski season", "6 months", "long-term")

For ANY other requirements (pet-friendly, balcony, parking, proximity to amenities, etc.), add them to an "additional_requirements" array.

Return ONLY valid JSON, no explanations."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract criteria from: {user_message}"}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        criteria_text = result['choices'][0]['message']['content']
        # Try to parse as JSON
        try:
            criteria = json.loads(criteria_text)
            return criteria
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', criteria_text, re.DOTALL)
            if json_match:
                criteria = json.loads(json_match.group(1))
                return criteria
            return {"error": "Failed to parse criteria as JSON"}
    
    except Exception as e:
        return {"error": str(e)}


def analyze_images(listing_content: str, max_images: int = 5) -> str:
    """Analyze listing images using OpenAI Vision API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Image analysis skipped (no API key)"
    
    # Extract image URLs from markdown - support multiple image formats
    pattern = r'!\[.*?\]\((https://[^\)]+\.(?:jpg|jpeg|png|webp))\)'
    urls = re.findall(pattern, listing_content, re.IGNORECASE)
    
    # If no markdown images found, try to extract raw image URLs
    if not urls:
        pattern_raw = r'https://[^\s<>"]+\.(?:jpg|jpeg|png|webp)'
        urls = re.findall(pattern_raw, listing_content, re.IGNORECASE)
    
    if not urls:
        return "No images found to analyze"
    
    # Limit images
    urls = list(set(urls))[:max_images]
    
    print(f"[Image Analysis] Found {len(urls)} unique images to analyze")
    
    analyses = []
    for idx, url in enumerate(urls):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this apartment image. Identify:
1. Room type
2. Key features and condition
3. Furnishing status
4. Notable amenities
5. Overall impression (scale 1-10)

Be concise but specific."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            # IMPORTANT: Include the URL so the LLM can extract it and display the image
            analyses.append(f"### Image {idx + 1}\n**Image URL:** {url}\n\n{analysis}\n\n---\n\n")
        except Exception as e:
            analyses.append(f"### Image {idx + 1}\n**Image URL:** {url}\n❌ Analysis failed: {str(e)}\n\n---\n\n")
    
    summary = "\n".join(analyses)
    print(f"[Image Analysis] Completed. Sample output: {summary[:300]}...")
    return summary


def generate_match_report(criteria: dict, listing_data: dict, image_analysis: str = "") -> str:
    """Generate the final match report using OpenAI"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Debug: Check what we're receiving
    print(f"[Debug generate_match_report] image_analysis length: {len(image_analysis) if image_analysis else 0}")
    print(f"[Debug generate_match_report] Has valid image analysis: {bool(image_analysis and image_analysis not in ['No images found to analyze', 'Image analysis skipped (no API key)'])}")
    
    system_prompt = """You are a helpful apartment rental advisor for the Swiss market. Your job is to analyze apartment listings and help users determine if they're a good match for their needs.

Be friendly, conversational, and encouraging. Extract all relevant details accurately from listings. Compare listings objectively against the user's specific criteria. Only evaluate criteria the user explicitly mentioned. Be realistic about "close enough" matches. Provide honest, actionable recommendations."""

    # Build the prompt
    prompt = f"""User's criteria:
```json
{json.dumps(criteria, indent=2)}
```

Listing data:
{listing_data.get('content', '')}

---

{f'''## Image Analysis Results:
{image_analysis}

**INSTRUCTION: You MUST include these image analysis results in a "📸 Photo Analysis" section in your output. Summarize the key findings from each image analysis above.**
''' if image_analysis and image_analysis not in ["No images found to analyze", "Image analysis skipped (no API key)"] else ''}

---

## Your Task:

Analyze this apartment listing and create a beautiful, user-friendly match report{" that includes the image analysis findings" if image_analysis and image_analysis not in ["No images found to analyze", "Image analysis skipped (no API key)"] else ""}.

### Output Format (use emojis and clear formatting):

```
# 🏠 Apartment Match Analysis

## 📋 Listing Summary
**Title:** [listing title]
**Location:** [full address/area]
**Price:** CHF [amount]/month
**Rooms:** [number] rooms
**Living Space:** [size] m²
**Available:** [date or immediately]

---

## 🎯 Match Score: [X]% 

[One sentence overall assessment]

---

## ✅ What Matches Your Criteria

[For EACH criterion that matches]

---

## ⚠️ Points to Consider

[For EACH criterion that doesn't match or is unclear]

---

## 💡 Key Highlights

• [Standout features]

---

{f'''## 📸 Photo Analysis

[Include the image analysis provided above. For each analyzed image, summarize the key findings about room type, condition, and features. Make it scannable and informative.]

---

''' if image_analysis and image_analysis not in ["No images found to analyze", "Image analysis skipped (no API key)"] else ''}

## 🤔 Our Recommendation

**[HIGHLY RECOMMENDED / WORTH CONSIDERING / NOT A GOOD FIT]**

[2-3 sentences explaining why]

---

## 📌 Next Steps

[Actionable steps]
```

{"IMPORTANT: If image analysis was provided above, you MUST include a '📸 Photo Analysis' section in your response that summarizes the image findings." if image_analysis and image_analysis not in ["No images found to analyze", "Image analysis skipped (no API key)"] else ""}

Return ONLY the formatted match analysis, ready to display to the user."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    # Debug: Print the prompt being sent (first 1000 chars)
    print(f"[Debug] Prompt being sent to LLM (first 1000 chars):\n{prompt[:1000]}")
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    except Exception as e:
        return f"Error generating match report: {str(e)}"


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message with apartment criteria and listing URL
    """
    try:
        # Extract URL from message
        user_message, listing_url = extract_url_from_message(request.message)
        
        if not listing_url:
            return ChatResponse(
                response="Please provide both your apartment criteria and a listing URL from Homegate.ch or similar sites.",
                status="error"
            )
        
        # Step 1: Extract user criteria
        criteria = extract_criteria_with_openai(user_message)
        if "error" in criteria:
            raise HTTPException(status_code=500, detail=f"Error extracting criteria: {criteria['error']}")
        
        # Step 2: Scrape listing
        listing_data = call_firecrawl_scraper(listing_url)
        if "error" in listing_data:
            raise HTTPException(status_code=500, detail=f"Error scraping listing: {listing_data['error']}")
        
        # Step 3: Analyze images (optional, can be skipped for speed)
        print(f"[Debug] Starting image analysis...")
        image_analysis = analyze_images(listing_data.get('content', ''), max_images=3)
        print(f"[Debug] Image analysis length: {len(image_analysis)}")
        print(f"[Debug] Image analysis result: {image_analysis[:500]}...")  # First 500 chars
        print(f"[Debug] Image analysis is valid: {image_analysis not in ['No images found to analyze', 'Image analysis skipped (no API key)']}")
        
        # Step 4: Generate match report
        print(f"[Debug] Generating match report with image_analysis={bool(image_analysis)}")
        match_report = generate_match_report(criteria, listing_data, image_analysis)
        
        return ChatResponse(
            response=match_report,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return f.read()


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
