from django.shortcuts import render, redirect
import os
import pymysql
import numpy as np
from numpy import dot
from numpy.linalg import norm
import re
import requests
import json

# -----------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------
uname = ""
logged_in = False

PIXABAY_API_KEY = "25167829-692ac73e994129337d29215fc"

VECTOR_SIZE = 300

# -----------------------------------------------------------------------
# TEXT CLEANING
# -----------------------------------------------------------------------
def cleanText(data):
    stop_words = {'a','an','the','and','or','but','in','on','at','to','for',
                  'of','with','by','from','is','was','are','were','be','been',
                  'have','has','had','do','does','did','will','would','could',
                  'should','may','might','shall','can','need','dare','ought',
                  'used','that','this','these','those','it','its'}
    data = data.split()
    data = [w for w in data if w.lower() not in stop_words]
    data = [word for word in data if len(word) > 3]
    return ' '.join(data)

# -----------------------------------------------------------------------
# ITINERARY GENERATOR (uses Claude API via Anthropic)
# -----------------------------------------------------------------------
def generate_itinerary_ai(source, destination, budget, description):
    """Call Anthropic Claude API to generate a detailed travel itinerary in Indian Rupees."""
    try:
        prompt = f"""Generate a comprehensive travel itinerary for a trip from {source} to {destination}.

Budget: ₹{budget} INR
User Preferences / Description: {description}

IMPORTANT:
- Use Indian Rupees only.
- Show all prices with ₹ symbol.
- Do not use dollars or USD anywhere.
- Total budget must be exactly ₹{budget}.
- Budget breakdown total must equal ₹{budget}.

Please provide a detailed travel plan with the following sections exactly in this order:

1. Introduction paragraph about {destination}
2. Assumptions
3. Budget Breakdown
4. Flights
5. Accommodation
6. Food
7. Transportation
8. Daily Itinerary
9. Important Notes
10. Travel Tips

Format the response clearly with each section header in ALL CAPS followed by a colon.
Make it detailed, practical, and specific to {destination}."""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]
            return text
        else:
            return None

    except Exception as e:
        print(f"Claude API error: {e}")
        return None


def generate_itinerary_fallback(source, destination, budget, description):
    """Fallback itinerary generator in Indian Rupees."""
    budget_int = int(budget) if str(budget).isdigit() else 200000

    flight_cost = int(budget_int * 0.35)
    hotel_cost = int(budget_int * 0.30)
    food_cost = int(budget_int * 0.15)
    transport_cost = int(budget_int * 0.10)
    misc_cost = budget_int - (flight_cost + hotel_cost + food_cost + transport_cost)

    return f"""INTRODUCTION:
Welcome to your personalized travel plan from {source} to {destination}! This itinerary is planned within your budget of ₹{budget_int}.

ASSUMPTIONS:
- Trip duration: 7 days / 6 nights
- Budget: ₹{budget_int} per person
- Travel season: Peak tourist season
- Accommodation: Budget/Mid-range hotels
- International flights included in budget
- Travel insurance recommended
- Based on your preferences: {description}

BUDGET BREAKDOWN:
- Flights (Round Trip): ₹{flight_cost}
- Accommodation (6 nights): ₹{hotel_cost}
- Food & Dining: ₹{food_cost}
- Local Transportation: ₹{transport_cost}
- Miscellaneous / Shopping: ₹{misc_cost}
- TOTAL: ₹{budget_int}

FLIGHTS:
Round-trip flights from {source} to {destination} are estimated around ₹{flight_cost}. Book tickets early to get better prices.

ACCOMMODATION:
Accommodation budget is ₹{hotel_cost} for 6 nights. Choose budget or mid-range hotels depending on the destination.

FOOD:
Food budget is ₹{food_cost}. Daily food budget estimate is approximately ₹{int(food_cost / 7)} per day.

TRANSPORTATION:
Local transportation budget is ₹{transport_cost}. Use metro, buses, local taxis, or rideshare apps where available.

DAILY ITINERARY:

Day 1 - Arrival:
Morning: Arrive at {destination}.
Afternoon: Check into hotel and rest.
Evening: Explore nearby local places.

Day 2 - City Tour:
Morning: Visit famous landmarks.
Afternoon: Explore museums or cultural places.
Evening: Try local food.

Day 3 - Nature / Sightseeing:
Morning: Visit natural attractions.
Afternoon: Photography and sightseeing.
Evening: Return to hotel.

Day 4 - Local Experience:
Morning: Visit markets.
Afternoon: Try local activities.
Evening: Relax and enjoy local cuisine.

Day 5 - Adventure / Shopping:
Morning: Adventure or sightseeing activity.
Afternoon: Shopping.
Evening: Visit a popular evening spot.

Day 6 - Day Trip:
Morning: Take a nearby day trip.
Afternoon: Explore nearby attractions.
Evening: Farewell dinner.

Day 7 - Departure:
Morning: Pack and checkout.
Afternoon: Travel to airport/railway station.
Evening: Return to {source}.

IMPORTANT NOTES:
- Check visa/passport requirements if traveling internationally.
- Carry valid ID proofs and travel documents.
- Keep emergency contacts saved.
- Carry some cash in local currency.
- Buy travel insurance.

TRAVEL TIPS:
- Book flights and hotels early.
- Compare prices before booking.
- Use public transport to save money.
- Keep digital and physical copies of documents.
- Follow local rules and culture."""


# -----------------------------------------------------------------------
# IMAGE SCRAPER (Pixabay)
# -----------------------------------------------------------------------
def search_images(query, num_images=5):
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={num_images}&safesearch=true"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "hits" in data:
            return [img['webformatURL'] for img in data['hits'][:num_images]]
        return []
    except Exception as e:
        print(f"Pixabay error: {e}")
        return []


def download_images(image_urls, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    downloaded = 0
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(f"{save_dir}/image_{i}.jpg", "wb") as f:
                f.write(response.content)
            downloaded += 1
        except Exception as e:
            print(f"Image download error: {e}")
    return downloaded


def scrapeImages(destination):
    """Download images for a destination if not already cached."""
    dest_clean = destination.strip().title()
    path = f'ItineraryApp/static/location_images/{dest_clean}'
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        os.makedirs(path, exist_ok=True)
        image_urls = search_images(dest_clean + " travel landmark", num_images=5)
        if image_urls:
            download_images(image_urls, path)


# -----------------------------------------------------------------------
# LOAD RAG DATA
# -----------------------------------------------------------------------
X = []
Y = []

def loadData():
    global X, Y
    feature_path = 'ItineraryApp/static/features/'
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs('ItineraryApp/static/model/', exist_ok=True)

    # Check dimension consistency
    if os.path.exists(feature_path + "X.npy"):
        try:
            old_X = np.load(feature_path + "X.npy", allow_pickle=True)
            if len(old_X.shape) > 1 and old_X.shape[1] != VECTOR_SIZE:
                os.remove(feature_path + "X.npy")
                if os.path.exists(feature_path + "Y.npy"):
                    os.remove(feature_path + "Y.npy")
        except Exception:
            pass

    if os.path.exists(feature_path + "X.npy") and os.path.exists(feature_path + "Y.npy"):
        try:
            X = np.load(feature_path + "X.npy", allow_pickle=True)
            Y = np.load(feature_path + "Y.npy", allow_pickle=True)
        except Exception:
            X = []
            Y = []

    flag = False
    model_dir = 'ItineraryApp/static/model'
    if not os.path.exists(model_dir):
        return

    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue
            y_list = list(Y) if len(Y) > 0 else []
            if file.lower() not in y_list:
                vector = np.random.rand(VECTOR_SIZE)
                if len(X) > 0:
                    x_list = X.tolist()
                    y_list = Y.tolist()
                else:
                    x_list = []
                    y_list = []
                x_list.append(vector)
                y_list.append(file.lower())
                X = np.array(x_list)
                Y = np.array(y_list)
                flag = True

    if flag:
        np.save(feature_path + "X.npy", X)
        np.save(feature_path + "Y.npy", Y)


# -----------------------------------------------------------------------
# VIEWS
# -----------------------------------------------------------------------
def index(request):
    request.session['logged_in'] = False
    request.session['username'] = ''
    return render(request, 'index.html', {})


def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def Register(request):
    return render(request, 'Register.html', {})


def TravelPlan(request):
    if not request.session.get('logged_in'):
        return redirect('/UserLogin.html')
    return render(request, 'TravelPlan.html', {})


def Logout(request):
    request.session['logged_in'] = False
    request.session['username'] = ''
    return redirect('/index.html')


# -----------------------------------------------------------------------
# LOGIN ACTION
# -----------------------------------------------------------------------
def UserLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', '').strip()
        password = request.POST.get('t2', '').strip()

        try:
            con = pymysql.connect(
                host='127.0.0.1', user='root', password='root', database='itinerary'
            )
            with con:
                cur = con.cursor()
                cur.execute(
                    "SELECT * FROM register WHERE username=%s AND password=%s",
                    (username, password)
                )
                result = cur.fetchone()

            if result:
                request.session['logged_in'] = True
                request.session['username'] = username
                return render(request, 'UserScreen.html', {
                    'data': f'<b>Welcome {username}!</b> You are now logged in. Click "Generate Travel Plan" to begin.',
                    'show_welcome': True
                })
            else:
                return render(request, 'UserLogin.html', {'data': 'Invalid username or password. Please try again.'})
        except Exception as e:
            return render(request, 'UserLogin.html', {'data': f'Database error: {str(e)}'})

    return redirect('/UserLogin.html')


# -----------------------------------------------------------------------
# REGISTER ACTION
# -----------------------------------------------------------------------
def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', '').strip()
        password = request.POST.get('t2', '').strip()
        contact = request.POST.get('t3', '').strip()
        email = request.POST.get('t4', '').strip()
        address = request.POST.get('t5', '').strip()

        try:
            con = pymysql.connect(
                host='127.0.0.1', user='root', password='root', database='itinerary'
            )
            with con:
                cur = con.cursor()
                cur.execute("SELECT username FROM register WHERE username=%s", (username,))
                if cur.fetchone():
                    return render(request, 'Register.html', {'data': 'Username already exists. Please choose a different username.'})

                cur.execute(
                    "INSERT INTO register VALUES (%s, %s, %s, %s, %s)",
                    (username, password, contact, email, address)
                )
                con.commit()

            return render(request, 'Register.html', {'data': 'Registration successful! Please login to continue.'})
        except Exception as e:
            return render(request, 'Register.html', {'data': f'Database error: {str(e)}'})

    return redirect('/Register.html')


# -----------------------------------------------------------------------
# TRAVEL PLAN ACTION
# -----------------------------------------------------------------------
def TravelPlanAction(request):
    if not request.session.get('logged_in'):
        return redirect('/UserLogin.html')

    if request.method == 'POST':
        source = request.POST.get('t1', '').strip()
        destination = request.POST.get('t2', '').strip()
        budget = request.POST.get('t3', '').strip()
        desc = request.POST.get('t4', '').strip()

        filename = f"{source}_{destination}_{budget}.txt"
        model_path = f"ItineraryApp/static/model/{filename}"

        os.makedirs("ItineraryApp/static/model", exist_ok=True)

        # Generate itinerary only if this exact file does not exist
        if not os.path.exists(model_path):
            plan = generate_itinerary_ai(source, destination, budget, desc)

            if not plan:
                plan = generate_itinerary_fallback(source, destination, budget, desc)

            with open(model_path, "w", encoding="utf-8") as f:
                f.write(plan)

        # Always read the exact file for entered source, destination, and budget
        with open(model_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Format plan as HTML
        result_html = format_itinerary_html(raw, source, destination, budget)

        # Scrape / load destination images
        dest_clean = destination.strip().title()
        img_dir = f"ItineraryApp/static/location_images/{dest_clean}"
        scrapeImages(destination)

        img_html = ""
        if os.path.exists(img_dir):
            imgs = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if imgs:
                img_html = (
                    '<div class="dest-images">'
                    '<h3>📸 ' + destination.title() + ' - Travel Gallery</h3>'
                    '<div class="img-row">'
                )

                for img in imgs[:5]:
                    img_html += (
                        f'<div class="img-cell">'
                        f'<img src="/static/location_images/{dest_clean}/{img}" '
                        f'alt="{destination} travel" />'
                        f'</div>'
                    )

                img_html += '</div></div>'

        return render(request, 'UserScreen.html', {
            'data': result_html,
            'images': img_html,
            'destination': destination.title(),
            'source': source.title(),
        })

    return redirect('/TravelPlan.html')

def format_itinerary_html(raw_text, source, destination, budget):
    """Convert raw itinerary text to styled HTML."""
    lines = raw_text.strip().split('\n')
    html = '<div class="itinerary-output">'
    html += f'<div class="itin-title">✈️ Travel Plan: {source.title()} → {destination.title()} | Budget: ₹{budget}</div>'

    section_keywords = [
        'INTRODUCTION', 'ASSUMPTIONS', 'BUDGET', 'FLIGHTS', 'ACCOMMODATION',
        'FOOD', 'TRANSPORTATION', 'DAILY ITINERARY', 'IMPORTANT NOTES',
        'TRAVEL TIPS', 'DAY 1', 'DAY 2', 'DAY 3', 'DAY 4', 'DAY 5',
        'DAY 6', 'DAY 7', 'NOTE', 'TIP'
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_section_header = False

        for kw in section_keywords:
            if stripped.upper().startswith(kw) and (stripped.endswith(':') or len(stripped) < 60):
                html += f'<div class="itin-section-header">{stripped}</div>'
                is_section_header = True
                break

        if not is_section_header:
            if stripped.startswith('-') or stripped.startswith('•'):
                html += f'<div class="itin-bullet">{stripped}</div>'
            elif stripped.startswith('Day ') or stripped.startswith('Morning') or stripped.startswith('Afternoon') or stripped.startswith('Evening'):
                html += f'<div class="itin-day">{stripped}</div>'
            else:
                html += f'<div class="itin-para">{stripped}</div>'

    html += '</div>'
    return html
