import gspread
from oauth2client.service_account import ServiceAccountCredentials
from playwright.sync_api import sync_playwright
import pandas as pd
import time
import os
import concurrent.futures
from threading import Lock
import threading
from queue import Queue
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

# Google Sheets API setup
def get_google_sheet(sheet_url, credentials_path):
    """Get data from Google Sheet using the third row as headers"""
    try:
        gc = gspread.service_account(filename=credentials_path)
        sheet = gc.open_by_url(sheet_url).sheet1
        
        # Fetch all values and use the third row as headers
        all_values = sheet.get_all_values()
        if len(all_values) < 3:
            print("Sheet does not have enough rows to determine headers.")
            return []
        
        # Use the third row as headers
        headers = all_values[2]
        records = [dict(zip(headers, row)) for row in all_values[3:]]
        
        return records
    except Exception as e:
        print(f"Error accessing sheet: {str(e)}")
        return []

# Function to scrape job postings using Playwright
def scrape_jobs(company_urls, keywords):
    job_listings = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Use Chromium in headless mode
        page = browser.new_page()

        for company in company_urls:
            url = company.get("Website", "")
            if not url:
                continue

            try:
                # Open the webpage using Playwright
                print(f"Scraping {url}...")
                page.goto(url)
                page_content = page.content().lower()  # Get the page content

                # Count matched keywords in job description
                matched_keywords = 0
                for keyword in keywords:
                    if keyword.lower() in page_content:
                        matched_keywords += 1

                # Calculate percentage of matched keywords
                matched_percentage = (matched_keywords / len(keywords)) * 100

                # If at least 50% of keywords match, save the job listing
                if matched_percentage >= 35:
                    job_listings.append({"Company": company["Company"], "URL": url, "Matched Percentage": matched_percentage})
                    print(f"Job listing at {url} matches {matched_percentage:.2f}% of your resume keywords.")

            except Exception as e:
                print(f"Error scraping {url}: {e}")

            time.sleep(1)  # Prevent being blocked

        browser.close()  # Close the browser

    return job_listings

class JobScraper:
    def __init__(self, keywords, max_workers=5):
        self.keywords = keywords
        self.max_workers = max_workers
        self.job_listings = []
        self.lock = Lock()
        self.progress_lock = Lock()
        self.processed_companies = 0
        self.total_companies = 0
        
    def update_progress(self):
        with self.progress_lock:
            self.processed_companies += 1
            progress = (self.processed_companies / self.total_companies) * 100
            print(f"Progress: {progress:.2f}% ({self.processed_companies}/{self.total_companies} companies processed)")

    def scrape_company(self, company):
        try:
            jobs = scrape_jobs([company], self.keywords)
            if jobs:
                with self.lock:
                    self.job_listings.extend(jobs)
            self.update_progress()
            return jobs
        except Exception as e:
            print(f"Error scraping {company}: {str(e)}")
            self.update_progress()
            return []

    def scrape_all_companies(self, companies):
        self.total_companies = len(companies)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.scrape_company, companies)
        return self.job_listings

def extract_skills_from_resume(resume_path, api_key):
    """Extract skills from resume using Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {api_key}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    
    # Read resume content
    resume_text = Path(resume_path).read_text(encoding='utf-8')
    
    # Skill categories
    skill_categories = [
        "programming_languages",
        "frameworks",
        "databases",
        "cloud_technologies",
        "tools_and_technologies"
    ]
    
    # API payload
    payload = {
        "inputs": resume_text,
        "parameters": {
            "candidate_labels": skill_categories,
            "multi_label": True
        }
    }
    
    # Make API call
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Add error handling and response debugging
    if response.status_code != 200:
        print(f"API Error: Status code {response.status_code}")
        print(f"Response: {response.text}")
        return []
        
    result = response.json()
    print(f"API Response: {result}")  # Debug print
    
    # Handle different response formats
    if isinstance(result, dict):
        labels = result.get('sequence_labels', result.get('labels', []))
        scores = result.get('sequence_scores', result.get('scores', []))
    elif isinstance(result, list) and len(result) > 0:
        labels = result[0].get('labels', [])
        scores = result[0].get('scores', [])
    else:
        print("Unexpected API response format")
        return []
    
    # Define keywords for each category
    skill_keywords = {
        "programming_languages": ["java", "python", "c++", "kotlin", "javascript", "typescript"],
        "frameworks": ["spring", "reactjs", "flask", "django", "net"],
        "databases": ["sql", "mysql", "postgresql", "mongodb", "redis"],
        "cloud_technologies": ["aws", "lambda", "s3", "ec2", "rds", "azure"],
        "tools_and_technologies": ["docker", "git", "figma", "GenAI", "llm", "ci/cd"]
    }
    
    # Extract relevant keywords based on top categories
    relevant_keywords = []
    for label, score in zip(labels, scores):
        if score > 0.3:  # Confidence threshold
            if label in skill_keywords:
                relevant_keywords.extend(skill_keywords[label])
    
    return list(set(relevant_keywords)) 

def main():
    print("Starting job search process...")
    start_time = time.time()
    
    SHEET_URL = "https://docs.google.com/spreadsheets/d/1-Be2li67qv0FWs_U2_fXTLnhWiYjPvvDZ94cTeNxSkg/edit#gid=0"
    RESUME_PATH = "/Users/admin/Desktop/google/resume.txt"
    CREDENTIALS_PATH = "/Users/admin/Desktop/google/arctic-acolyte-451122-s2-e8fc9d28a3fa.json"  # Update this path
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_TOKEN")  # Added quotes around environment variable name
    
    print("Extracting skills from resume...")
    if not HUGGINGFACE_API_KEY.startswith("hf_"):
        print("Warning: Invalid Hugging Face API key format. Get your key from https://huggingface.co/settings/tokens")
        keywords = ["python", "java", "javascript"]  # Fallback keywords
    else:
        keywords = extract_skills_from_resume(RESUME_PATH, HUGGINGFACE_API_KEY)
    
    print(f"Using keywords: {keywords}")
    
    print("Fetching companies from Google Sheet...")
    companies = get_google_sheet(SHEET_URL, CREDENTIALS_PATH)
    
    if not companies:
        print("No companies found in sheet. Please check your credentials and sheet URL.")
        return
        
    print(f"Found {len(companies)} companies")
    
    print("Starting job scraping...")
    scraper = JobScraper(keywords, max_workers=5)
    job_listings = scraper.scrape_all_companies(companies)
    
    if job_listings:
        df = pd.DataFrame(job_listings)
        df.to_csv("filtered_job_listings.csv", index=False)
        print(f"\nJob scraping complete! Found {len(job_listings)} matching positions.")
        print(f"Results saved to 'filtered_job_listings.csv'")
    else:
        print("\nNo job listings found.")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
