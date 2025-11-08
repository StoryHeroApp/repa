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

Extraction Rules:
1. If "for X persons/people" → use "occupants": X
2. If "ski season" or temporary → use "duration": "ski season" or appropriate period
3. If "price is not a problem" or "budget flexible" → do NOT include min_rent or max_rent
4. If "more than X rooms" → use "min_rooms": X
5. If "less than CHF Y" → use "max_rent": Y
6. If "about X square meters" → use both "min_living_space" and "max_living_space" with ±10% range
7. Location can be specific (city/postal code) OR proximity-based ("close to ski", "near train station")
8. Extract EACH specific requirement as a separate item in additional_requirements
9. Preserve the user's exact wording and intent
10. Infer room requirements from occupancy if helpful (e.g., 5 persons might suggest larger apartment)
11. Return ONLY valid JSON, no explanations

Example 1 - Full numeric criteria:
Input: "I am looking for an apartment in 8008 Zürich, more than 4 rooms, living space about 100 square meters, and rent less than CHF 5000."
Output:
{
  "location": "8008 Zürich",
  "min_rooms": 4,
  "min_living_space": 90,
  "max_living_space": 110,
  "max_rent": 5000
}

Example 2 - Seasonal/proximity focused:
Input: "I'm visiting Switzerland for a ski season and need an apartment for 5 persons, need it to be super close to the ski action. Price is not a problem."
Output:
{
  "occupants": 5,
  "duration": "ski season",
  "location": "ski resort area",
  "additional_requirements": ["close to ski slopes", "ski-in/ski-out preferred", "suitable for 5 people"]
}

Example 3 - Mixed criteria:
Input: "Looking for 3 rooms in Zürich, max CHF 3000, with parking space, balcony, and modern kitchen"
Output:
{
  "location": "Zürich",
  "min_rooms": 3,
  "max_rent": 3000,
  "additional_requirements": ["parking space", "balcony", "modern kitchen"]
}

Example 4 - Only location:
Input: "I need an apartment in Bern"
Output:
{
  "location": "Bern"
}

Now extract the criteria:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    user_prompt = f"""Now extract the criteria from the User's Request:
<user_request>
{user_message}
</user_request>"""
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
                            "text": """Analyze this apartment/property image. Identify:
1. Room type (living room, bedroom, kitchen, bathroom, exterior, view, etc.)
2. Key features and condition (modern, renovated, spacious, natural light, etc.)
3. Furnishing status (furnished, unfurnished, partially furnished)
4. Notable amenities or highlights
5. Overall impression (scale 1-10)

Be concise but specific. Focus on details that would matter to a renter."""
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

## Your Approach:
- Be friendly, conversational, and encouraging
- Extract all relevant details accurately from listings
- Compare listings objectively against the user's specific criteria
- Only evaluate criteria the user explicitly mentioned - don't penalize for unspecified requirements
- Be realistic about "close enough" matches (e.g., 95m² ≈ 100m², Zürich City ≈ 8008 Zürich)
- Distinguish between deal-breakers and nice-to-haves
- Provide honest, actionable recommendations

## Swiss Rental Context:
- Understand Swiss room counting (e.g., 3.5 rooms = 2 bedrooms + living room + half room)
- Know typical Zürich pricing and neighborhoods
- Recognize common Swiss rental terms and amenities
- Consider public transport accessibility and local area quality

## Tone:
- Professional yet warm
- Clear and direct
- Helpful and supportive
- Honest about both positives and concerns

Follow the exact output format provided in the user's request."""

    # Determine if we have images to display
    has_images = image_analysis and image_analysis not in ["No images found to analyze", "Image analysis skipped (no API key)"]
    
    image_analysis_section = ""
    if has_images:
        image_analysis_section = f"""## Image Analysis Results:
{image_analysis}
"""
    
    image_gallery_section = ""
    if has_images:
        image_gallery_section = """## 📸 Photo Analysis

**INSTRUCTION:** Extract all image URLs from the Image Analysis section and create a beautiful photo gallery here. For each analyzed image:
1. Display the image using: ![Room Name](image_url)
2. Add a brief caption based on the analysis

Example format:
### Living Room
![Living Room](https://media2.homegate.ch/.../image1.jpg)
*Modern, spacious living area with natural light and contemporary furnishing.*

### Kitchen
![Kitchen](https://media2.homegate.ch/.../image2.jpg)
*Fully equipped kitchen with modern appliances and ample counter space.*

Continue for all analyzed images..."""
    
    with_images = " with photo gallery" if has_images else ""

    # Build the user prompt
    prompt = f"""User's criteria:
```json
{json.dumps(criteria, indent=2)}
```

Listing data:
<listing>
{listing_data.get('content', '')}
</listing>

{image_analysis_section}

---

## Your Task:

Analyze this apartment listing and create a beautiful, user-friendly match report{with_images}.

**CRITICAL IMAGE INSTRUCTION:** 
The listing data contains a **LISTING_IMAGE_URL:** field. You MUST extract the COMPLETE URL (do not truncate it) and insert it at the very top of your response using this EXACT format:
![Apartment](COMPLETE_URL_HERE)

Make sure to copy the entire URL exactly as provided, including all characters after the last slash.

### Output Format (use emojis and clear formatting):

```
# 🏠 Apartment Match Analysis

![Apartment](INSERT_COMPLETE_LISTING_IMAGE_URL_HERE)

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

[For EACH criterion that matches, use this format:]
**✓ [Criterion Name]**
• Your requirement: [what user asked for]
• Listing offers: [what listing has]
• Assessment: [brief positive note]

---

## ⚠️ Points to Consider

[For EACH criterion that doesn't match or is unclear:]
**⚠ [Criterion Name]**
• Your requirement: [what user asked for]
• Listing offers: [what listing has]
• Impact: [why this matters - deal-breaker or negotiable?]

[If no concerns: *No significant concerns - all criteria met!*]

---

## 💡 Key Highlights

• [Standout feature 1]
• [Standout feature 2]
• [Standout feature 3]
• [Other notable amenities]

---

{image_gallery_section}

---

## 🤔 Our Recommendation

**[HIGHLY RECOMMENDED / WORTH CONSIDERING / NOT A GOOD FIT]**

[2-3 sentences explaining why, considering the user's priorities and the listing's strengths/weaknesses. Be honest and helpful.]

---

## 📌 Next Steps

[If recommended: Suggest they contact the landlord, schedule viewing, etc.]
[If not recommended: Suggest what to look for instead]

---

[ONLY IF HIGHLY RECOMMENDED OR WORTH CONSIDERING:]

## ✉️ Personalized Contact Message

Ready to send! Copy this message for the "Contact Advertiser" form on Homegate.ch:

---
**Subject:** Interest in [Room count]-Room Apartment at [Location]

Dear Sir/Madam,

I am writing to express my strong interest in the [room count]-room apartment at [address/location] listed for CHF [price]/month.

[Include 2-3 sentences about why this apartment is perfect for them based on their criteria - be specific! Reference actual matches like "The 105m² living space and location in 8008 Zürich are exactly what I've been searching for."]

About me:
• [Infer likely tenant profile based on their search - e.g., "Professional working in Zürich" or "Small family" based on room requirements]
• Reliable, non-smoking tenant with excellent references
• Available to move in [reference availability date from listing or say "immediately"]
• Long-term rental desired

I am very interested in scheduling a viewing at your earliest convenience. I am flexible with timing and can meet this week if possible.

I have prepared all necessary documents (employment contract, salary statements, references) and am ready to proceed quickly given the competitive Zürich rental market.

Looking forward to hearing from you.

Best regards,
[Your Name]
[Your Phone]
[Your Email]
---

**Tip:** Personalize further by adding:
- Your current situation (relocating, growing family, etc.)
- Why you chose this specific listing
- Your move-in timeline
- Any relevant lifestyle details (quiet, respectful neighbor, etc.)

Good apartments in Zürich get many applications - send this today! ⚡
```

### Important Instructions:
1. **Be conversational and friendly** - write like you're helping a friend
2. **Use emojis** to make it visually appealing and scannable
3. **Be honest** - if something doesn't match, say so clearly
4. **Prioritize** - focus on what matters most (deal-breakers vs nice-to-haves)
5. **Only compare specified criteria** - don't penalize for unspecified requirements
6. **Extract all listing details** - even if not in criteria (they're useful to see)
7. **Be realistic** - 95m² is close enough to 100m², Zürich City ≈ 8008 Zürich
8. **Consider Swiss context** - room counting, pricing norms, etc.
9. **Make it actionable** - give clear next steps
10. **Generate contact message ONLY for recommended listings** - skip this section if "NOT A GOOD FIT"
11. **Personalize the contact message** based on the user's actual criteria matches (be specific about what matched!)
12. **Make the contact message professional yet warm** - increase their chances in competitive market

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
