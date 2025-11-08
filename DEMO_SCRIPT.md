# REPA Demo Script (90 seconds)

## Setup (Before Demo)
- Have browser open to: **https://repa.onrender.com** (or localhost:8000)
- Have this example ready to paste:
```
I'm visiting Switzerland for ski season and want to hit the slopes! I got my wife and her family coming along. So it will be 3-4 people. We want to be right close to the ski action! Check this listing: https://www.homegate.ch/rent/4002583790
```

---

## Demo Script

### Introduction (15 seconds)
*"Hi everyone! Today I'm showing you REPA - Real Estate Personalized Assistant. It's an AI-powered tool that analyzes Swiss rental listings and tells you if they match your criteria."*

### The Problem (10 seconds)
*"Finding the perfect apartment in Switzerland is time-consuming. You have to read through dozens of listings, compare details, and figure out if they meet your needs."*

### The Workflow (15 seconds)
*[Show LangFlow diagram on screen]*

*"Here's how REPA works behind the scenes:"*
1. **User inputs criteria + listing URL**
2. **AI extracts criteria** into structured JSON
3. **Firecrawl scrapes** the listing page
4. **Vision AI analyzes** property images
5. **AI matches** criteria against listing
6. **Generates personalized report** with recommendation

### Live Demo (40 seconds)

*"Let me show you in action."*

**[Paste the example query and hit Send]**

*"I'm looking for a ski season rental for 3-4 people, close to the slopes. Let me check this listing..."*

**[Wait 10-15 seconds while processing]**

*"And here's the result! REPA gives me:"*
- ✅ **Match score** (30% - not a good fit)
- 📋 **Complete listing summary** - price, rooms, location
- ✅ **What matches** - ski-in/ski-out, great views
- ⚠️ **Issues** - only fits 3 people max, not 4
- 📸 **Photo analysis** - shows actual images with AI insights
- 💡 **Recommendation** - NOT A GOOD FIT due to capacity
- 📌 **Next steps** - suggestions on what to look for

*"Notice it even analyzed the apartment photos and included them in the report!"*

### Closing (10 seconds)
*"That's REPA! In seconds, you get a comprehensive match analysis instead of spending 15 minutes reading through listings. Perfect for Swiss apartment hunting!"*

**[Optional: Show the LangFlow diagram again]**

*"Built with OpenAI GPT-4, Firecrawl for scraping, and deployed on Render. Questions?"*

---

## Key Points to Emphasize
1. **Speed** - Analysis in ~15 seconds vs 15 minutes manual review
2. **AI Vision** - Analyzes actual apartment photos
3. **Honest Recommendations** - Tells you when something is NOT a good fit
4. **Swiss Market Focus** - Understands Swiss rental terminology and context
5. **Production Ready** - Live on Render, integrated with real APIs

## Backup Talking Points (If Time)
- "The LLM dynamically extracts criteria - you can describe needs in natural language"
- "Vision AI analyzes room types, furnishing, condition, and amenities"
- "Even generates a personalized contact message for good matches"
- "Built entirely from a LangFlow workflow converted to FastAPI"

---

## Technical Stack (If Asked)
- **Frontend**: Vanilla HTML/CSS/JS with Marked.js for markdown
- **Backend**: FastAPI (Python)
- **AI**: OpenAI GPT-4o-mini for text + vision
- **Scraping**: Firecrawl API
- **Deployment**: Render (auto-deploy from GitHub)
- **Origin**: Converted from LangFlow visual workflow
