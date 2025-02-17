# ResumeMatcherBot

ResumeMatcherBot is an intelligent job scraping tool that automates job searching by matching job descriptions with your resume. It uses **Playwright** to extract job postings, compares them against your resume using a similarity algorithm, and filters the most relevant opportunities. Integrated with **Google Sheets**, it provides direct links to best-matched jobs.

## Features
- **Automated Job Scraping**: Uses Playwright to navigate and extract job postings.
- **Resume Matching**: Compares job descriptions with your resume using a similarity algorithm.
- **Google Sheets Integration**: Reads company URLs and job postings from Google Sheets.
- **CSV Export**: Saves filtered job listings for easy access.
- **Customizable Keywords**: Allows filtering by specific technologies and skills.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Google Sheets API credentials (JSON key file)
- Playwright with Chromium

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ResumeMatcherBot.git
   cd ResumeMatcherBot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   playwright install
   ```
3. Configure Google Sheets API:
   - Place your JSON key file in the project directory.
   - Update the script to use your **Google Sheets URL**.

## Usage
1. Add company URLs to your Google Sheet.
2. Run the script:
   ```sh
   python main.py
   ```
3. The script will scrape job descriptions, match them against your resume, and save the best matches in `filtered_job_listings.csv`.

## Configuration
- **Google Sheets URL**: Update `SHEET_URL` in `main.py` with your sheet link.
- **Keywords**: Modify `KEYWORDS` to match your skills and expertise.

## Contributing
Feel free to submit pull requests or report issues.

## License
MIT License

---
Developed by Atif Ali Khan

