import gspread
from oauth2client.service_account import ServiceAccountCredentials
from playwright.sync_api import sync_playwright
import pandas as pd
import time
import os

# Google Sheets API setup
def get_google_sheet(sheet_url):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.environ.get('GOOGLE_CREDENTIALS_PATH'), scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).sheet1
    return sheet.get_all_records()

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

# Main execution
def main():
    SHEET_URL = "https://docs.google.com/spreadsheets/d/1-Be2li67qv0FWs_U2_fXTLnhWiYjPvvDZ94cTeNxSkg/edit#gid=0"
    KEYWORDS = ["java", "python", "c++", "kotlin", "sql", "mysql", "postgresql", "mongodb", "reactjs", "css3", "html5", "javascript", "typescript", "graphql", "restful", "spring", "net", "flask", "django", "aws", "lambda", "s3", "ec2", "rds", "API", "iot core", "api gateway", "redis", "kafka", "docker", "GenAI", "active directory", "git", "figma","llm", "models","ci/cd","azure"]

    companies = get_google_sheet(SHEET_URL)
    job_listings = scrape_jobs(companies, KEYWORDS)

    if job_listings:
        df = pd.DataFrame(job_listings)
        df.to_csv("filtered_job_listings.csv", index=False)
        print("Job scraping complete! Results saved to 'filtered_job_listings.csv'")
    else:
        print("No job listings found.")

if __name__ == "__main__":
    main()
