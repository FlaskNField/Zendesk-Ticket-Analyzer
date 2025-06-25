import os
import requests
import pandas as pd
import openai
import json
import re
import spacy
import argparse
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Load environment variables from .env file
load_dotenv()

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Kapa.ai Configuration
KAPA_API_KEY = os.getenv('KAPA_API_KEY')
KAPA_PROJECT_ID = os.getenv('KAPA_PROJECT_ID')
KAPA_INTEGRATION_ID = os.getenv('KAPA_INTEGRATION_ID')

# Load credentials from environment variables
ZENDESK_API_TOKEN = os.getenv('ZENDESK_API_TOKEN')
ZENDESK_EMAIL = os.getenv('ZENDESK_EMAIL')
ZENDESK_SUBDOMAIN = os.getenv('ZENDESK_SUBDOMAIN')
ZENDESK_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"

# OpenAI Configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

# --- Configuration Mappings ---
TICKET_FORM_IDS = {
    'account-billing': 360003888493,
    'technical-support': 14917317203995
}
DATE_RANGE_MAP = {
    '24hr': 1,
    'last-week': 7,
    'last-month': 30
}
# -----------------------------

# --- Color Palette ---
COLOR_NO = '#f42a45'
COLOR_YES = '#00e65e'
PIE_COLORS = [COLOR_YES, COLOR_NO]  # Yes, No
# -----------------------------

KAPA_ERROR_LOG = "kapa_errors.log"

def log_kapa_error(block: str):
    with open(KAPA_ERROR_LOG, "a") as f:
        f.write(block + "\n")

def check_environment_variables():
    """Check if all required environment variables are set."""
    required_vars = [
        'ZENDESK_API_TOKEN',
        'ZENDESK_EMAIL',
        'ZENDESK_SUBDOMAIN',
        'OPENAI_API_KEY',
        'KAPA_API_KEY',
        'KAPA_PROJECT_ID',
        'KAPA_INTEGRATION_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set these variables in your environment or .env file."
        )

# Retry configuration for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_zendesk_tickets(days_to_look_back, ticket_form_id):
    """Fetch tickets from Zendesk created in the last day that are solved or closed."""
    start_date = datetime.now() - timedelta(days=days_to_look_back)
    query = f"created>={start_date.strftime('%Y-%m-%d')} status:solved status:closed ticket_form_id:{ticket_form_id}"
    
    url = f"{ZENDESK_URL}/search.json"
    assert ZENDESK_EMAIL is not None and ZENDESK_API_TOKEN is not None
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
    params = {"query": query}
    
    response = requests.get(url, auth=auth, params=params)
    response.raise_for_status()
    return response.json()['results']

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ticket_comments(ticket_id):
    """Fetch all comments for a specific ticket."""
    url = f"{ZENDESK_URL}/tickets/{ticket_id}/comments.json"
    assert ZENDESK_EMAIL is not None and ZENDESK_API_TOKEN is not None
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
    
    response = requests.get(url, auth=auth)
    response.raise_for_status()
    return response.json()['comments']

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_ticket_content(ticket_content):
    """Analyze ticket content using OpenAI's GPT-4 to get summary and self-service analysis."""
    system_prompt = (
        "You are a support ticket analyzer. Analyze the ticket and respond in JSON format. "
        "The JSON object must contain these keys: "
        "'summary' (a brief summary of the customer's issue and its resolution), "
        "'self_service_possible' ('Yes' or 'No'), "
        "'self_service_reason' (a brief explanation), and "
        "'self_service_recommendation' (if self-service was not possible, suggest an improvement "
        "like 'Create new KB article', 'Update docs', or 'Product enhancement'. If it was possible, respond with 'N/A'). "
        "Format the self-service analysis as follows: "
        "Self Service Possible?: <Yes/No>\n\nReason:\n<reason>"
    )
    user_prompt = f"Please analyze this support ticket:\n\n{ticket_content}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        analysis_data = json.loads(response.choices[0].message.content)
        return analysis_data
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from OpenAI response."}
    except Exception as e:
        return {"error": f"Error analyzing ticket content: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_themes(ticket_summaries):
    """Analyze a batch of ticket summaries to identify and assign themes."""
    system_prompt = (
        "You are a support trends analyst. Based on the provided list of ticket summaries, "
        "first, identify between 5 and 10 high-level themes. "
        "Then, assign one of your defined themes to each ticket. "
        "Respond with a single JSON object where the keys are the ticket IDs (as strings) "
        "and the values are the assigned theme."
    )
    user_prompt = f"Here are the ticket summaries:\n\n{json.dumps(ticket_summaries, indent=2)}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        themes_data = json.loads(response.choices[0].message.content)
        return themes_data
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from OpenAI theme analysis."}
    except Exception as e:
        return {"error": f"Error analyzing themes: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_kapa_response(question: str) -> str:
    """Query Kapa.ai to get a suggested answer."""
    if not all([KAPA_API_KEY, KAPA_PROJECT_ID, KAPA_INTEGRATION_ID]):
        return "Kapa.ai API key, Project ID, or Integration ID not configured."
    url = f"https://api.kapa.ai/query/v1/projects/{KAPA_PROJECT_ID}/chat/"
    headers = {
        'X-API-KEY': KAPA_API_KEY,
        'Content-Type': 'application/json'
    }
    data = {
        'query': question,
        'integration_id': KAPA_INTEGRATION_ID
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 400:
            block = ("\n--- KAPA ERROR DEBUG INFO (HTTP 400) ---\n"
                     f"Request URL: {url}\n"
                     f"Request Headers: {headers}\n"
                     f"Request Payload: {json.dumps(data, indent=2)}\n"
                     f"Response Content: {response.text}\n"
                     "--- END KAPA ERROR DEBUG INFO ---\n")
            print(block)
            log_kapa_error(block)
            return "Kapa API request was invalid (HTTP 400). Please check your Kapa project settings or try again later."
        response.raise_for_status()
        response_json = response.json()
        return response_json.get('answer', 'N/A - Answer key not found in Kapa.ai response')
    except requests.exceptions.HTTPError as e:
        block = ("\n--- KAPA ERROR DEBUG INFO (HTTPError) ---\n"
                 f"Request URL: {url}\n"
                 f"Request Headers: {headers}\n"
                 f"Request Payload: {json.dumps(data, indent=2)}\n"
                 f"Response Content: {e.response.text if hasattr(e, 'response') and e.response is not None else 'N/A'}\n"
                 "--- END KAPA ERROR DEBUG INFO ---\n")
        print(block)
        log_kapa_error(block)
        return f"Error from Kapa.ai API: {e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'} {e.response.text if hasattr(e, 'response') and e.response is not None else 'N/A'}"
    except requests.exceptions.RequestException as e:
        block = ("\n--- KAPA ERROR DEBUG INFO (RequestException) ---\n"
                 f"Request URL: {url}\n"
                 f"Request Headers: {headers}\n"
                 f"Request Payload: {json.dumps(data, indent=2)}\n"
                 f"Exception: {str(e)}\n"
                 "--- END KAPA ERROR DEBUG INFO ---\n")
        print(block)
        log_kapa_error(block)
        return f"Error querying Kapa.ai: {str(e)}"
    except json.JSONDecodeError:
        block = ("\n--- KAPA ERROR DEBUG INFO (JSONDecodeError) ---\n"
                 f"Request URL: {url}\n"
                 f"Request Headers: {headers}\n"
                 f"Request Payload: {json.dumps(data, indent=2)}\n"
                 f"Response Content: {response.text if 'response' in locals() else 'N/A'}\n"
                 "--- END KAPA ERROR DEBUG INFO ---\n")
        print(block)
        log_kapa_error(block)
        return "Error: Could not decode Kapa.ai response."

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def compare_answers(kapa_answer: str, agent_answer: str):
    """Uses OpenAI to compare Kapa's answer with the agent's answer."""
    system_prompt = (
        "You are an answer evaluator. Compare the AI-generated answer to the human agent's answer for the same customer query. "
        "Respond in JSON with two keys: 'did_kapa_get_it_right' (a string, 'Yes' or 'No') and "
        "'kapa_correctness_reason' (a brief explanation for your decision)."
    )
    user_prompt = (
        f"Kapa's Answer:\n{kapa_answer}\n\n"
        f"Agent's Answer:\n{agent_answer}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        comparison_data = json.loads(response.choices[0].message.content)
        return comparison_data
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from OpenAI comparison."}
    except Exception as e:
        return {"error": f"Error comparing answers: {str(e)}"}

def wrap_text(text, length=100):
    """Manually wraps text by inserting newlines."""
    if not isinstance(text, str):
        return text
    
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > length:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word
    lines.append(current_line)
    return '\n'.join(lines).strip()

def show_drilldown_window(filtered_df, title, zendesk_subdomain, columns_to_show):
    """Creates a new window to display a list or table of tickets."""
    drilldown_window = tk.Toplevel()
    drilldown_window.title(title)
    drilldown_window.geometry("4800x1000") # Even larger window for tables

    def open_link(url):
        webbrowser.open_new(url)

    # --- If only showing ticket numbers, use a simple text list ---
    if len(columns_to_show) == 1 and columns_to_show[0] == 'Ticket #':
        scrollbar = ttk.Scrollbar(drilldown_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget = tk.Text(drilldown_window, wrap=tk.WORD, padx=10, pady=10, yscrollcommand=scrollbar.set)
        text_widget.pack(expand=True, fill=tk.BOTH)
        scrollbar.config(command=text_widget.yview)
        text_widget.tag_configure("hyperlink", foreground="#0066cc", underline=True)
        text_widget.tag_bind("hyperlink", "<Enter>", lambda e: text_widget.config(cursor="hand2"))
        text_widget.tag_bind("hyperlink", "<Leave>", lambda e: text_widget.config(cursor=""))

        for i, ticket_id in enumerate(filtered_df['Ticket #']):
            url = f"https://{zendesk_subdomain}.zendesk.com/agent/tickets/{ticket_id}"
            tag = f"link-{i}"
            text_widget.insert(tk.END, f"{ticket_id}\n", ("hyperlink", tag))
            text_widget.tag_bind(tag, "<Button-1>", lambda e, u=url: open_link(u))
        text_widget.config(state=tk.DISABLED)
        return

    # --- Otherwise, create a detailed table using Treeview ---
    style = ttk.Style(drilldown_window)
    style.configure("Treeview", rowheight=80, font=('Calibri', 10)) # Taller rows for wrapped text
    style.configure("Treeview.Heading", font=('Calibri', 11, 'bold'))
    
    tree_frame = ttk.Frame(drilldown_window)
    tree_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    tree = ttk.Treeview(tree_frame, columns=columns_to_show, show='headings')
    
    # Define columns and headings with wrapping
    for col in columns_to_show:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor='w')

    # Add data to the tree, with wrapping
    for index, row in filtered_df.iterrows():
        wrapped_values = [wrap_text(row[col]) for col in columns_to_show]
        tree.insert("", tk.END, values=wrapped_values)

    def on_tree_click(event):
        item_id = tree.identify_row(event.y)
        if item_id:
            ticket_id = tree.item(item_id)['values'][0] # Assuming Ticket # is always the first column
            url = f"https://{zendesk_subdomain}.zendesk.com/agent/tickets/{ticket_id}"
            open_link(url)

    tree.bind("<Double-1>", on_tree_click)
    
    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
    hsb.pack(side='bottom', fill='x')
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    tree.pack(expand=True, fill=tk.BOTH)

def format_self_service_analysis(possible, reason):
    return f"Self Service Possible?: {possible}\n\nReason:\n{reason}"

def launch_summary_gui(csv_file_path, date_range_label, zendesk_subdomain, args):
    """Launches a Tkinter GUI to display the analysis summary."""
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return

    # --- Data Processing ---
    total_tickets = len(df)
    
    # --- GUI Setup ---
    root = tk.Tk()
    root.title("Zendesk Ticket Analysis Summary")
    root.geometry("900x700")

    # Use a more universal font
    DEFAULT_FONT = ("Helvetica", 11)
    HEADER_FONT = ("Helvetica", 14, "bold")
    SUBHEADER_FONT = ("Helvetica", 13, "bold")
    BOLD_FONT = ("Helvetica", 12, "bold")
    ITALIC_FONT = ("Helvetica", 11, "italic")

    # --- Scrollable main frame ---
    container = ttk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
    vscrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscrollbar.set)
    vscrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    main_frame = ttk.Frame(canvas, padding="10")
    main_frame_id = canvas.create_window((0, 0), window=main_frame, anchor="nw")
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    main_frame.bind("<Configure>", on_configure)
    # Make mousewheel scroll the canvas
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # --- Navigation Stack ---
    view_stack = []

    def clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def show_dashboard():
        clear_frame(main_frame)
        # --- Header ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=5)
        ttk.Label(header_frame, text=f"Total Tickets Analyzed: {total_tickets}", font=HEADER_FONT).pack(side=tk.LEFT)
        ttk.Label(header_frame, text=f"Timeframe: {date_range_label.replace('-', ' ').title()}", font=HEADER_FONT).pack(side=tk.RIGHT)
        # --- Deflection Rating ---
        deflection = get_kapa_deflection(total_tickets)
        ttk.Label(header_frame, text=f"Ticket Deflection (Kapa/Total): {deflection}", font=SUBHEADER_FONT, foreground="#4E79A7").pack(side=tk.LEFT, padx=20)
        # --- Charts ---
        charts_frame = ttk.Frame(main_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        # Create two frames for two rows of pie charts
        charts_row1 = ttk.Frame(charts_frame)
        charts_row1.pack(anchor='w', pady=5)
        charts_row2 = ttk.Frame(charts_frame)
        charts_row2.pack(anchor='w', pady=5)
        # First row: Self-Service and Kapa
        create_pie_chart(charts_row1, df, 'Self-Service Analysis', "Self-Service Possible?", ['Ticket #', 'Theme', 'Self-Service Analysis'])
        create_pie_chart(charts_row1, df, 'Did Kapa Get It Right?', "Did Kapa Get It Right?", ['Ticket #', 'Customer Issue', 'Did Kapa Get It Right?', 'Kapa Correctness Reason'])
        # Second row: Sentiment and Deflection
        create_pie_chart(charts_row2, df, 'Sentiment', "Customer Sentiment Breakdown", ['Ticket #', 'Sentiment'])
        sentiment_counts = df['Sentiment'].value_counts()
        show_deflection_pie(sentiment_counts, sentiment_counts.index.tolist(), ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"], parent=charts_row2)
        # --- Themes List ---
        themes_frame = ttk.Frame(main_frame)
        themes_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        ttk.Label(themes_frame, text="Ticket Themes Breakdown (Double-click to drill down)", font=BOLD_FONT).pack(anchor='w')
        if 'Theme' not in df.columns or df['Theme'].dropna().empty:
            ttk.Label(themes_frame, text="No themes found for this selection.", font=ITALIC_FONT, foreground="#888").pack(pady=10, anchor='w')
        else:
            theme_counts = df['Theme'].value_counts().reset_index()
            theme_counts.columns = ['Theme', 'Count']
            theme_tree = ttk.Treeview(themes_frame, columns=['Theme', 'Count'], show='headings')
            theme_tree.heading('Theme', text='Theme')
            theme_tree.heading('Count', text='Ticket Count')
            theme_tree.column('Theme', width=400)
            theme_tree.column('Count', width=100, anchor='center')
            for index, row in theme_counts.iterrows():
                theme_tree.insert("", tk.END, values=[row['Theme'], row['Count']], tags=('theme-row',))
            theme_tree.tag_configure('highlight', background='#E8F0FE')
            def on_theme_hover(event):
                item = theme_tree.identify_row(event.y)
                for child in theme_tree.get_children():
                    theme_tree.item(child, tags=('theme-row',))
                if item:
                    theme_tree.item(item, tags=('highlight',))
            def on_theme_leave(event):
                for child in theme_tree.get_children():
                    theme_tree.item(child, tags=('theme-row',))
            theme_tree.bind("<Motion>", on_theme_hover)
            theme_tree.bind("<Leave>", on_theme_leave)
            def on_theme_click(event):
                item_id = theme_tree.identify_row(event.y)
                if item_id:
                    selected_theme = theme_tree.item(item_id)['values'][0]
                    filtered_df = df[df['Theme'] == selected_theme]
                    window_title = f"Tickets for Theme: {selected_theme}"
                    show_drilldown(filtered_df, window_title, ['Ticket #', 'Theme'])
            theme_tree.bind("<Double-1>", on_theme_click)
            theme_tree.pack(anchor='w', padx=10, pady=5)

    def show_drilldown(filtered_df, title, columns_to_show):
        clear_frame(main_frame)
        # Back button
        back_frame = ttk.Frame(main_frame)
        back_frame.pack(fill=tk.X, pady=5)
        def go_back():
            show_dashboard()
        back_btn = ttk.Button(back_frame, text="← Back", command=go_back)
        back_btn.pack(side=tk.LEFT, padx=5)
        # Main drilldown frame (table + details)
        drilldown_frame = ttk.Frame(main_frame)
        drilldown_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        # Treeview: only show Ticket # in parent row
        tree = ttk.Treeview(drilldown_frame, columns=['Ticket #'], show='headings', selectmode='browse')
        tree.heading('Ticket #', text='Ticket #')
        tree.column('Ticket #', width=200, anchor='w')
        vsb = ttk.Scrollbar(drilldown_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(drilldown_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        drilldown_frame.grid_rowconfigure(0, weight=1)
        drilldown_frame.grid_columnconfigure(0, weight=1)
        # Insert ticket numbers
        ticket_id_to_row = {}
        for idx, row in filtered_df.iterrows():
            item_id = tree.insert('', 'end', values=[row['Ticket #']])
            ticket_id_to_row[item_id] = row
        # Details panel below
        details_frame = ttk.Frame(drilldown_frame)
        details_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=(10,0))
        details_text = tk.Text(details_frame, wrap=tk.WORD, height=20, font=("Helvetica", 12))
        details_text.pack(expand=True, fill=tk.BOTH)
        details_text.config(state=tk.DISABLED)
        # Configure tag for headings and hyperlink
        details_text.tag_configure('heading', font=("Helvetica", 14, "bold"))
        details_text.tag_configure('hyperlink', foreground="#0066cc", underline=True)
        def open_ticket_url(ticket_number):
            url = f"https://{zendesk_subdomain}.zendesk.com/agent/tickets/{ticket_number}"
            webbrowser.open_new(url)
        # When a ticket is selected, show details with section headings and hyperlink
        def on_tree_select(event):
            selected = tree.focus()
            if not selected or selected not in ticket_id_to_row:
                details_text.config(state=tk.NORMAL)
                details_text.delete(1.0, tk.END)
                details_text.config(state=tk.DISABLED)
                return
            row = ticket_id_to_row[selected]
            details_text.config(state=tk.NORMAL)
            details_text.delete(1.0, tk.END)
            # Insert Ticket # as hyperlink at the top
            ticket_number = row['Ticket #']
            url = f"https://{zendesk_subdomain}.zendesk.com/agent/tickets/{ticket_number}"
            details_text.insert(tk.END, f"Ticket #: ", 'heading')
            start_idx = details_text.index(tk.INSERT)
            details_text.insert(tk.END, f"{ticket_number}\n", 'hyperlink')
            end_idx = details_text.index(tk.INSERT)
            details_text.tag_add('ticket_link', start_idx, end_idx)
            def click_link(event, ticket_number=ticket_number):
                open_ticket_url(ticket_number)
            details_text.tag_bind('ticket_link', '<Button-1>', click_link)
            # Insert the rest of the details (all columns except Ticket #)
            for col in row.index:
                if col == 'Ticket #':
                    continue
                details_text.insert(tk.END, f"{col}\n", 'heading')
                details_text.insert(tk.END, f"{row[col]}\n\n")
            details_text.config(state=tk.DISABLED)
        tree.bind('<<TreeviewSelect>>', on_tree_select)
        # Remove single-click open; use double-click instead
        def on_tree_click(event):
            item = tree.identify_row(event.y)
            if not item or item not in ticket_id_to_row:
                return
            ticket_number = ticket_id_to_row[item]['Ticket #']
            open_ticket_url(ticket_number)
        # Change to double-click
        tree.unbind('<ButtonRelease-1>')
        tree.bind('<Double-1>', on_tree_click)

    def create_pie_chart(frame, data_df, column, title, drilldown_cols):
        # Check if column exists
        if column not in data_df.columns:
            ttk.Label(frame, text=f"Column '{column}' not found in data.", font=ITALIC_FONT, foreground="#888").pack()
            return
        chart_container = ttk.Frame(frame, width=350, height=300)
        chart_container.pack_propagate(False)
        chart_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, anchor='n')
        fig = Figure(figsize=(3.5, 3), dpi=100)
        ax = fig.add_subplot(111)
        if column == 'Self-Service Analysis':
            chart_data = data_df[column].str.extract(r'Self Service Possible\?: (Yes|No)').iloc[:, 0].value_counts()
            chart_data = chart_data[chart_data.index.isin(['Yes', 'No'])]
            color_map = [COLOR_YES if label == 'Yes' else COLOR_NO for label in chart_data.index]
        elif column == 'Did Kapa Get It Right?':
            chart_data = data_df[column].value_counts()
            chart_data = chart_data[chart_data.index.isin(['Yes', 'No'])]
            color_map = [COLOR_YES if label == 'Yes' else COLOR_NO for label in chart_data.index]
        elif column == 'Sentiment':
            chart_data = data_df[column].value_counts()
            base_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
            color_map = (base_colors * ((len(chart_data) // len(base_colors)) + 1))[:len(chart_data)]
        else:
            chart_data = data_df[column].value_counts()
            color_map = None
        if chart_data.empty:
            ttk.Label(chart_container, text=f"No data for '{column}'.", font=ITALIC_FONT, foreground="#888").pack()
            return
        def autopct_format(pct):
            total = sum(chart_data)
            count = int(round(pct * total / 100.0))
            return f"{count} ({pct:.1f}%)" if pct > 0 else ''
        wedges, _, _ = ax.pie(chart_data, autopct=autopct_format, startangle=90, colors=color_map, labels=chart_data.index)
        ax.set_title(title)
        for wedge in wedges:
            wedge.set_picker(True)
            wedge.set_label(wedge.get_label())
            wedge.original_edgecolor = wedge.get_edgecolor()
            wedge.original_linewidth = wedge.get_linewidth()
        def on_pick(event):
            wedge = event.artist
            label = wedge.get_label()
            if column == 'Self-Service Analysis':
                filtered_df = data_df[data_df[column].str.contains(f"Self Service Possible\?: {label}", na=False)]
            else:
                filtered_df = data_df[data_df[column] == label]
            window_title = f"{title}: {label} ({len(filtered_df)} tickets)"
            if column == 'Sentiment':
                show_drilldown(filtered_df, window_title, ['Ticket #', 'Sentiment'])
            elif column == 'Did Kapa Get It Right?':
                show_drilldown(filtered_df, window_title, ['Ticket #', 'Customer Issue', 'Did Kapa Get It Right?', 'Kapa Correctness Reason'])
            elif column == 'Self-Service Analysis':
                show_drilldown(filtered_df, window_title, ['Ticket #', 'Theme', 'Self-Service Analysis'])
            else:
                show_drilldown(filtered_df, window_title, drilldown_cols)
        highlighted_wedge = None
        def on_hover(event):
            nonlocal highlighted_wedge
            if event.inaxes == ax:
                for wedge in wedges:
                    if wedge.contains(event)[0]:
                        if wedge != highlighted_wedge:
                            if highlighted_wedge:
                                highlighted_wedge.set_edgecolor(highlighted_wedge.original_edgecolor)
                                highlighted_wedge.set_linewidth(highlighted_wedge.original_linewidth)
                            wedge.set_edgecolor('yellow')
                            wedge.set_linewidth(2.5)
                            highlighted_wedge = wedge
                        fig.canvas.draw_idle()
                        return
            if highlighted_wedge:
                highlighted_wedge.set_edgecolor(highlighted_wedge.original_edgecolor)
                highlighted_wedge.set_linewidth(highlighted_wedge.original_linewidth)
                highlighted_wedge = None
                fig.canvas.draw_idle()
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        canvas = FigureCanvasTkAgg(fig, master=chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_deflection_pie(deflection_data, deflection_labels, deflection_colors, parent):
        fig = Figure(figsize=(3.5, 3), dpi=100)
        ax = fig.add_subplot(111)
        def autopct_format(pct):
            total = sum(deflection_data)
            count = int(round(pct * total / 100.0))
            return f"{count} ({pct:.1f}%)" if pct > 0 else ''
        wedges, _, _ = ax.pie(deflection_data, autopct=autopct_format, startangle=90, colors=deflection_colors, labels=deflection_labels)
        ax.set_title("Ticket Deflection Breakdown")
        # Hover glow effect
        for wedge in wedges:
            wedge.set_picker(True)
            wedge.set_label(wedge.get_label())
            wedge.original_edgecolor = wedge.get_edgecolor()
            wedge.original_linewidth = wedge.get_linewidth()
        highlighted_wedge = None
        def on_hover(event):
            nonlocal highlighted_wedge
            if event.inaxes == ax:
                for wedge in wedges:
                    if wedge.contains(event)[0]:
                        if wedge != highlighted_wedge:
                            if highlighted_wedge:
                                highlighted_wedge.set_edgecolor(highlighted_wedge.original_edgecolor)
                                highlighted_wedge.set_linewidth(highlighted_wedge.original_linewidth)
                            wedge.set_edgecolor('yellow')
                            wedge.set_linewidth(2.5)
                            highlighted_wedge = wedge
                        fig.canvas.draw_idle()
                        return
            if highlighted_wedge:
                highlighted_wedge.set_edgecolor(highlighted_wedge.original_edgecolor)
                highlighted_wedge.set_linewidth(highlighted_wedge.original_linewidth)
                highlighted_wedge = None
                fig.canvas.draw_idle()
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    show_dashboard()
    root.mainloop()

def redact_pii(text):
    """Redacts common PII from text using spaCy and regex."""
    # First, use regex for structured PII like emails and phones
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[REDACTED EMAIL]', text)
    text = re.sub(r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED PHONE]', text)
    
    # Then, use spaCy for contextual PII like names and organizations
    doc = nlp(text)
    new_text = list(text)
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "ORG"]:
            label = f"[REDACTED {ent.label_}]"
            start = ent.start_char
            end = ent.end_char
            new_text[start:end] = label
    
    return "".join(new_text)

def print_ascii_art():
    """Prints Baltimore-themed ASCII art."""
    crab = r"""
     /\
    ( /   @ @    ()
     \\ __| |__  /
      \/   "   \/
     /-|       |-\
    / /-\     /-\ \
     / /-`---'-\ \
      /         \ 
    """
    print(crab)
    print("      David Larsen")
    print("\n")

def main():
    """Main execution function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Analyze Zendesk tickets using AI.")
    parser.add_argument(
        '--ticket-type',
        type=str,
        choices=TICKET_FORM_IDS.keys(),
        default='account-billing',
        help=f"The type of Zendesk ticket to analyze. Defaults to 'account-billing'."
    )
    parser.add_argument(
        '--date-range',
        type=str,
        choices=DATE_RANGE_MAP.keys(),
        default='24hr',
        help="The time period to analyze. Defaults to '24hr'."
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help="Skip launching the graphical dashboard (for non-interactive environments)."
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help="Path to a previously generated CSV file to use for the dashboard. If not provided, the latest zendesk_analysis_*.csv will be used."
    )
    args = parser.parse_args()

    days_to_look_back = DATE_RANGE_MAP[args.date_range]
    ticket_form_id = TICKET_FORM_IDS[args.ticket_type]
    zendesk_subdomain = os.getenv('ZENDESK_SUBDOMAIN')

    print_ascii_art()
    print(f"Starting analysis for '{args.ticket_type}' tickets from the last {args.date_range}...")

    # If --csv is provided or a recent CSV exists, use it
    if args.csv:
        csv_file = args.csv
        print(f"Using provided CSV file: {csv_file}")
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return
        output_file = csv_file
        # Use the date range from the filename if possible
        date_range_label = args.date_range
        launch_summary_gui(output_file, date_range_label, zendesk_subdomain, args)
        return
    else:
        csv_files = sorted(glob.glob("zendesk_analysis_*.csv"), reverse=True)
        if csv_files:
            output_file = csv_files[0]
            print(f"Using latest CSV file: {output_file}")
            date_range_label = args.date_range
            launch_summary_gui(output_file, date_range_label, zendesk_subdomain, args)
            return

    def process_ticket(ticket):
        ticket_id = ticket['id']
        kapa_suggestion = 'N/A'  # Default value
        comparison_result = {'did_kapa_get_it_right': 'N/A', 'kapa_correctness_reason': 'N/A'}
        recommendation = ''
        try:
            comments = get_ticket_comments(ticket_id)
            # Find the first public comment from the original requester for Kapa.ai
            requester_id = ticket.get('requester_id')
            first_customer_comment = None
            if requester_id:
                for comment in comments:
                    if comment.get('author_id') == requester_id and comment.get('public', True):
                        body = comment.get('plain_body', '').strip()
                        if body:
                            first_customer_comment = body
                            break
            if first_customer_comment:
                redacted_question = redact_pii(first_customer_comment)
                kapa_suggestion = get_kapa_response(redacted_question)
                kapa_suggestion = redact_pii(kapa_suggestion) # Sanitize Kapa's output as a precaution
                # Sentiment analysis
                sentiment, sentiment_explanation = analyze_sentiment(redacted_question)
            else:
                kapa_suggestion = 'N/A - No customer comment found.'
                sentiment, sentiment_explanation = 'N/A', 'N/A'
            # Find the last public comment from an agent (assignee) to get the resolution
            assignee_id = ticket.get('assignee_id')
            agent_resolution = None
            if assignee_id:
                for comment in reversed(comments):
                    if comment.get('author_id') == assignee_id and comment.get('public', True):
                        body = comment.get('plain_body', '').strip()
                        if body:
                            agent_resolution = body
                            break
            # Compare answers if we have both
            if agent_resolution and kapa_suggestion and not kapa_suggestion.startswith('N/A'):
                redacted_agent_resolution = redact_pii(agent_resolution)
                raw_comparison = compare_answers(kapa_suggestion, redacted_agent_resolution)
                if "error" not in raw_comparison:
                    comparison_result = {
                        'did_kapa_get_it_right': raw_comparison.get('did_kapa_get_it_right', 'Error'),
                        'kapa_correctness_reason': raw_comparison.get('kapa_correctness_reason', 'Error parsing reason')
                    }
                else:
                    comparison_result['kapa_correctness_reason'] = raw_comparison['error']
            elif not agent_resolution:
                comparison_result['kapa_correctness_reason'] = 'N/A - No agent resolution found.'
            ticket_content = "\n".join([
                f"Comment by {comment.get('author_id')} at {comment.get('created_at')}:\n{redact_pii(comment.get('plain_body', ''))}"
                for comment in comments
            ])
            if not ticket_content.strip():
                analysis_data = {"summary": "Ticket has no content.", "error": "No content"}
            else:
                analysis_data = analyze_ticket_content(ticket_content)
            if "error" in analysis_data:
                summary = analysis_data.get("summary", "")
                self_service_analysis = f"Error processing ticket: {analysis_data['error']}"
                possible = 'N/A'
            else:
                summary = analysis_data.get('summary', 'N/A')
                possible = analysis_data.get('self_service_possible', 'N/A')
                reason = analysis_data.get('self_service_reason', 'N/A')
                recommendation_text = analysis_data.get('self_service_recommendation', 'N/A')
                self_service_analysis = format_self_service_analysis(possible, reason)
                if possible == 'No':
                    self_service_analysis += f"\nRecommendation: {recommendation_text}"
            # Only generate recommendation if self-service is not possible
            if possible == 'No':
                recommendation = get_ticket_recommendation(
                    summary,
                    self_service_analysis,
                    kapa_suggestion,
                    comparison_result.get('did_kapa_get_it_right', 'N/A')
                )
            return {
                'Ticket #': ticket_id,
                'Assignee': ticket.get('assignee_id', 'Unassigned'),
                'Date Created': ticket['created_at'],
                'Analysis': summary,
                'Self-Service Analysis': self_service_analysis,
                'Kapa.ai Suggestion': kapa_suggestion,
                'Did Kapa Get It Right?': comparison_result['did_kapa_get_it_right'],
                'Kapa Correctness Reason': comparison_result['kapa_correctness_reason'],
                'Theme': ticket.get('theme', ''),
                'Recommendation': recommendation,
                'Sentiment': sentiment,
                'Sentiment Explanation': sentiment_explanation,
                'Customer Issue': redact_pii(first_customer_comment) if first_customer_comment else ''
            }
        except Exception as e:
            return {
                'Ticket #': ticket_id,
                'Assignee': ticket.get('assignee_id', 'Unassigned'),
                'Date Created': ticket.get('created_at', 'Unknown'),
                'Analysis': f"Error processing ticket: {str(e)}",
                'Self-Service Analysis': 'N/A',
                'Kapa.ai Suggestion': kapa_suggestion,
                'Did Kapa Get It Right?': comparison_result['did_kapa_get_it_right'],
                'Kapa Correctness Reason': comparison_result['kapa_correctness_reason'],
                'Theme': ticket.get('theme', ''),
                'Recommendation': recommendation,
                'Sentiment': 'N/A',
                'Sentiment Explanation': 'N/A',
                'Customer Issue': ''
            }

    try:
        check_environment_variables()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"zendesk_analysis_{timestamp}.csv"
        tickets = get_zendesk_tickets(days_to_look_back, ticket_form_id)
        # --- Parallel ticket processing ---
        if not tickets:
            results = []
        else:
            results = [None] * len(tickets)
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_idx = {executor.submit(process_ticket, ticket): idx for idx, ticket in enumerate(tickets)}
                for future in tqdm(as_completed(future_to_idx), total=len(tickets), desc="Analyzing Tickets"):
                    idx = future_to_idx[future]
                    try:
                        if 0 <= idx < len(results):
                            results[idx] = future.result()
                        else:
                            results.append(future.result())
                    except Exception as e:
                        if 0 <= idx < len(results):
                            results[idx] = {'Ticket #': tickets[idx]['id'], 'Analysis': f'Error: {e}'}
                        else:
                            results.append({'Ticket #': tickets[idx]['id'], 'Analysis': f'Error: {e}'})
        if results:
            ticket_summaries = {str(res['Ticket #']): res['Analysis'] for res in results if 'Error' not in res['Analysis']}
            if ticket_summaries:
                themes = analyze_themes(ticket_summaries)
                if "error" not in themes:
                    for res in results:
                        res['Theme'] = themes.get(str(res['Ticket #']), 'N/A')
                else:
                    for res in results:
                        res['Theme'] = f"Error: {themes['error']}"

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Analysis complete. Results saved to {output_file}")
        if not args.no_gui:
            try:
                if input("Launch graphical dashboard? (y/n): ").lower().strip() == 'y':
                    # Launch the GUI
                    launch_summary_gui(output_file, args.date_range, zendesk_subdomain, args)
            except EOFError:
                print("Non-interactive environment detected. Skipping GUI launch.")
        else:
            print("GUI launch skipped due to --no-gui flag.")
        
    except ImportError as e:
        print(f"ImportError: {e}\nPlease ensure all dependencies are installed. Run:\n  pip install pandas openai spacy tenacity python-dotenv matplotlib tqdm\n  python -m spacy download en_core_web_md")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ticket_recommendation(summary, self_service_analysis, kapa_suggestion, kapa_correctness):
    """Get an AI-powered recommendation for next steps based on the ticket analysis."""
    prompt = (
        "You are a support operations expert. Based on the following ticket summary, self-service analysis, and Kapa (AI chatbot) performance, "
        "recommend the most impactful next step. Choose from: show an existing Zendesk article when the customer opens a ticket, create a new Zendesk article, teach Kapa about something specific, request a product update, or another actionable step. "
        "Be concise and actionable.\n\n"
        f"Ticket Summary: {summary}\n\n"
        f"Self-Service Analysis: {self_service_analysis}\n\n"
        f"Kapa Suggestion: {kapa_suggestion}\n\n"
        f"Kapa Correctness: {kapa_correctness}\n\n"
        "Recommendation:"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting recommendation: {str(e)}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_sentiment(customer_message):
    """Analyze customer sentiment using OpenAI and return label and explanation."""
    system_prompt = '''You are a customer support sentiment analyst.
Analyze the following customer support ticket and assign it one of the following sentiment labels:
- Positive
- Neutral
- Frustrated
- Angry
- Delighted

Return your response in this structured format:

Sentiment: <one of the 5 labels>
Explanation: <1-2 sentences explaining why you chose this sentiment based on tone, language, or content of the message>

Customer Message:
"""
[CUSTOMER MESSAGE GOES HERE]
"""'''
    user_prompt = f'''Customer Message:
"""
{customer_message}
"""'''
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300
        )
        content = response.choices[0].message.content
        import re
        sentiment_match = re.search(r"Sentiment:\s*(.+)", content)
        explanation_match = re.search(r"Explanation:\s*(.+)", content)
        sentiment = sentiment_match.group(1).strip() if sentiment_match else 'N/A'
        explanation = explanation_match.group(1).strip() if explanation_match else 'N/A'
        return sentiment, explanation
    except Exception as e:
        return 'N/A', f'Error analyzing sentiment: {str(e)}'

def get_kapa_deflection(total_tickets):
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=1)
        start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = f"https://api.kapa.ai/query/v1/projects/{KAPA_PROJECT_ID}/analytics/activity/?start_date_time={start_str}&end_date_time={end_str}&aggregation_period=day"
        headers = {'X-API-KEY': KAPA_API_KEY} if KAPA_API_KEY else {}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        kapa_convos = data.get('aggregate', {}).get('total_conversations', 0)
        if total_tickets == 0:
            return 'N/A'
        return f"{kapa_convos}/{total_tickets} ({(kapa_convos/total_tickets)*100:.1f}%)"
    except Exception as e:
        return f"Error: {e}"

# Add a function to fetch Zendesk article views for the timeframe
def get_zendesk_article_views(start_time):
    try:
        import math
        api_token = os.getenv('ZENDESK_API_TOKEN')
        subdomain = os.getenv('ZENDESK_SUBDOMAIN')
        if not api_token or not subdomain:
            print("Zendesk API token or subdomain not set. Cannot fetch article views.")
            return 0
        url = f"https://{subdomain}.zendesk.com/api/v2/help_center/article_view_events.json?start_time={start_time}"
        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        count = 0
        next_page = url
        page_count = 0
        MAX_PAGES = 20
        while next_page and page_count < MAX_PAGES:
            resp = requests.get(next_page, headers=headers, timeout=30)
            if resp.status_code == 404:
                print(f"Zendesk article view endpoint not found. Check your subdomain, permissions, and Guide plan.")
                return 0
            resp.raise_for_status()
            data = resp.json()
            events = data.get('article_view_events', [])
            count += len(events)
            next_page = data.get('next_page')
            page_count += 1
        if page_count == MAX_PAGES:
            print("Warning: Reached max page limit when fetching article views.")
        return count
    except Exception as e:
        print(f"Error fetching Zendesk article views: {e}")
        return 0

# Add a function to get Kapa conversation count for the timeframe
def get_kapa_convo_count(start_time):
    try:
        now = datetime.now(timezone.utc)
        end_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
        start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = f"https://api.kapa.ai/query/v1/projects/{KAPA_PROJECT_ID}/analytics/activity/?start_date_time={start_str}&end_date_time={end_str}&aggregation_period=day"
        headers = {'X-API-KEY': KAPA_API_KEY} if KAPA_API_KEY else {}
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 400:
            print(f"Kapa API returned 400 Bad Request. Check your project ID and date range.")
            return 0
        resp.raise_for_status()
        data = resp.json()
        return data.get('aggregate', {}).get('total_conversations', 0)
    except Exception as e:
        print(f"Error fetching Kapa conversations: {e}")
        return 0

if __name__ == "__main__":
    main() 
