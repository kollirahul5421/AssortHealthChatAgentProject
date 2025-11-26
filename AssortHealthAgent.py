import os
from dotenv import load_dotenv
import requests
from openai import OpenAI
load_dotenv()

#Functional Requirements:
# Will be using prompt engineering to guide the agent's behavior.
# The agent will be able to gather patient information from the user and set up an appointment with them.
#1. Gather patient information from the user (Full name, Date of birth)
#2 Gather insurance information from the user (Insurance company, Insurance ID)
#3 Gather Medical information: (Reason behind seaking care/ Reason for visit)
#4 Address information: Validate using Google Maps API
#5 Allow for appointment selection
#   - present available providers and times
#   - user can select their preferred appointment
#6 At the end display a summary that includes:
#   - selected appointment time/date
#   - assgined phsyician
#   - All collected information

SYSTEM_PROMPT = """
You are a patient-intake assistant for a healthcare clinic.

Your job is to collect patient information step-by-step before the patient sees a clinician.

Required Information (in order):
1. Patient Information:
   - Full name
   - Date of birth (MM/DD/YYYY)
   - Address (street, city, state, zip code) — validate using Google Maps API

2. Insurance Information:
   - Insurance company name
   - Insurance ID (optional — ask only if the user indicates they have it)

3. Medical Information:
   - Reason for visit (chief complaint)

4. Appointment Selection:
   - Present available time slots to the patient, e.g., 
     "1) 10:00 AM Monday
      2) 11:30 AM Tuesday
      3) 2:00 PM Wednesday"
   - Ask the patient to choose one by entering the corresponding number
   - Confirm the chosen time
   - If the patient enters an invalid number, ask again

Interaction Rules:
1. You must ask ONLY ONE question at a time.
2. Follow the order of required information exactly.
3. If the users answer is unclear or incomplete, ask a clarifying question.
4. If a user says they do not have insurance, skip the insurance ID step.
5. Do NOT provide medical advice or diagnosis.
6. Be friendly and welcoming. 
7. After collecting all required fields, respond **only** in a readable summary format with all collected information, including:
   - Selected appointment time/date
   - Assigned physician
   - Full patient intake information (name, DOB, address, insurance, reason for visit)
8. End the conversation with a friendly confirmation message like:
   "Your appointment is confirmed. Have a nice day!"

9. If any field is missing, continue the conversation and ask only for the missing field (one at a time).

Begin the conversation with this welcome message:

"Hi there! I'm here to help with your check-in today. I'll ask you a few quick questions so we can get your information to the care team. Let's get started — what's your full name?"
"""

# STATE (Conversation Memory)
state = {
    "step": "NAME",
    "full_name": None,
    "date_of_birth": None,
    "address": None,
    "insurance_company": None,
    "insurance_id": None,
    "reason_for_visit": None,
    "appointment_time": None,
    "assigned_physician": "Dr. Smith"
}

# Create OpenAI Client
def get_client():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)

# Validate Address Using Google Maps Geocoding API
def validate_address(street, city, state, zip_code):
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    api_key = os.getenv(GOOGLE_MAPS_API_KEY)

    address = f"{street}, {city}, {state} {zip_code}"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] != "OK" or len(data["results"]) == 0:
        return False, None  # invalid address

    # Take the first result
    result = data["results"][0]
    formatted_address = result["formatted_address"]
    location = result["geometry"]["location"]

    return True, {"formatted_address": formatted_address, "lat": location["lat"], "lng": location["lng"]}

# Validate Appointment Choice
def get_appointment_choice(user_input, available_slots):
    """
    Check if user_input is a valid numbered selection from available_slots.
    Returns (True, selected_slot) if valid, (False, None) if invalid.
    """
    try:
        choice = int(user_input.strip())
        if 1 <= choice <= len(available_slots):
            return True, available_slots[choice - 1]
        else:
            return False, None
    except ValueError:
        return False, None

# Chat Logic (Main Function)
def chat(message, history, state):
    client = get_client()

    # ADDRESS STEP — Validate the user's entry
    if state["step"] == "ADDRESS":
        # You can split message into street, city, state, zip or assume comma-separated
        parts = [p.strip() for p in message.split(",")]
        if len(parts) < 4:
            return "Please provide your full address in the format: street, city, state, zip code.", history

        valid, info = validate_address(*parts)
        if not valid:
            return "Sorry, we couldn’t verify that address. Can you enter it again, including street, city, state, and zip code?", history

        state["address"] = info["formatted_address"]
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Got it! Verified address: {info['formatted_address']}"})

        state["step"] = "INSURANCE"
        return "Next, what is your insurance company name?", history
    
    # Hard-coded these available slots but can adjust if needed
    available_slots = ["10:00 AM Monday", "11:30 AM Tuesday", "2:00 PM Wednesday"]

    # APPOINTMENT STEP — Validate selection
    if state["step"] == "APPOINTMENT":
        valid, slot = get_appointment_choice(message, available_slots)
        if valid:
            state["appointment_time"] = slot
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Great! Your appointment is set for {slot}."})
            state["step"] = "COMPLETE"

            # Generate summary
            summary = {
                "full_name": state.get("full_name"),
                "date_of_birth": state.get("date_of_birth"),
                "address": state.get("address"),
                "insurance_company": state.get("insurance_company"),
                "insurance_id": state.get("insurance_id"),
                "reason_for_visit": state.get("reason_for_visit"),
                "appointment_time": state.get("appointment_time"),
                "assigned_physician": state.get("assigned_physician", "Dr. Smith")
            }

            summary_message = (
                "✅ Patient Intake Complete! Here’s a summary of your information:\n\n"
                f"- Full Name: {summary['full_name']}\n"
                f"- Date of Birth: {summary['date_of_birth']}\n"
                f"- Address: {summary['address']}\n"
                f"- Insurance: {summary['insurance_company']} ({summary['insurance_id']})\n"
                f"- Reason for Visit: {summary['reason_for_visit']}\n"
                f"- Selected Appointment Time: {summary['appointment_time']}\n"
                f"- Assigned Physician: {summary['assigned_physician']}\n\n"
                "Your appointment is confirmed. Have a nice day! 😊"
            )


            history.append({"role": "assistant", "content": summary_message})
            return summary_message, history
        else:
            return f"Sorry, that is not a valid choice. Please select a number from 1 to {len(available_slots)}.", history


    # All other steps handled by LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": message})

    #Error logging if client fails to connect
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
    except Exception as e:
        print("RAW ERROR:", e)
        return "I'm having trouble right now—can we try again soon?", history

    bot_message = response.choices[0].message.content
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_message})

    return bot_message, history

# Console Application Entry Point
if __name__ == "__main__":
    history = []
    state = {"step": "NAME", "full_name": None, "date_of_birth": None,
             "address": None, "insurance_company": None, "insurance_id": None,
             "reason_for_visit": None}

    print("Chatbot started! Type 'quit' to exit.\n")
    print("Bot: Hi there! I’m here to help with your check-in today. I’ll ask you a few quick questions so we can get your information to the care team. Let’s get started — what’s your full name?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response, history = chat(user_input, history, state)
        print("Bot:", response)


    

















