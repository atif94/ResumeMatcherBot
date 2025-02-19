import gspread
from oauth2client.service_account import ServiceAccountCredentials
from playwright.sync_api import sync_playwright, Browser, Page
import pandas as pd
import time
import os
import concurrent.futures
from threading import Lock
import threading
from queue import Queue
from pathlib import Path
import requests
import json
import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("job_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Google Sheets API setup
def get_google_sheet(sheet_url, credentials_path):
    """Get data from Google Sheet using the third row as headers"""
    try:
        gc = gspread.service_account(filename=credentials_path)
        sheet = gc.open_by_url(sheet_url).sheet1
        
        # Fetch all values and use the third row as headers
        all_values = sheet.get_all_values()
        if len(all_values) < 3:
            logger.error("Sheet does not have enough rows to determine headers.")
            return []
        
        # Use the third row as headers
        headers = all_values[2]
        records = [dict(zip(headers, row)) for row in all_values[3:]]
        
        logger.info(f"Successfully loaded {len(records)} records from Google Sheet")
        return records
    except Exception as e:
        logger.error(f"Error accessing sheet: {str(e)}")
        return []

class NERSkillExtractor:
    """Extract skills from resume using Named Entity Recognition"""
    
    def __init__(self, use_api: bool = True, api_key: Optional[str] = None):
        """
        Initialize the skill extractor
        
        Args:
            use_api: Whether to use the Hugging Face API (True) or local spaCy model (False)
            api_key: Hugging Face API key if using the API
        """
        self.use_api = use_api
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/spacy/en_pipeline_ner_trf"
        
        # Common skill terms and technology keywords
        self.tech_skills = self._load_tech_skills()
        
        # Load spaCy model if not using API
        if not use_api:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_trf")
                logger.info("Loaded spaCy transformer model")
            except OSError:
                logger.warning("Transformer model not found, attempting to download...")
                import spacy.cli
                spacy.cli.download("en_core_web_trf")
                self.nlp = spacy.load("en_core_web_trf")
                logger.info("Downloaded and loaded spaCy transformer model")
            except ImportError:
                logger.error("spaCy not installed. Please install it with: pip install spacy")
                raise
    
    def _load_tech_skills(self) -> Dict[str, List[str]]:
        """Load technology skills database"""
        return {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang", "ruby", 
                "php", "swift", "kotlin", "rust", "scala", "r", "matlab", "perl", "shell", "bash",
                "powershell", "html", "css", "sql", "nosql", "dart", "julia"
            ],
            "frameworks_libraries": [
                "react", "angular", "vue", "django", "flask", "spring", "spring boot", "express", 
                "node.js", "next.js", "svelte", "jquery", "bootstrap", "tailwind", "tensorflow", 
                "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
                "d3.js", "react native", "flutter", "electron", "symfony", "laravel", ".net",
                "rails", "xamarin", "fastapi", "gin", "echo", "phoenix", "catalyst", "deno"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "cassandra", "redis", "oracle", "sql server",
                "sqlite", "dynamodb", "couchbase", "firebase", "elasticsearch", "neo4j", "mariadb",
                "cockroachdb", "bigtable", "cosmosdb", "snowflake", "teradata", "hbase", "influxdb"
            ],
            "cloud_technologies": [
                "aws", "amazon web services", "ec2", "s3", "lambda", "azure", "google cloud", 
                "gcp", "firebase", "heroku", "digitalocean", "cloudflare", "netlify", "vercel",
                "kubernetes", "docker", "openshift", "serverless", "terraform", "cloudformation",
                "ecs", "eks", "fargate", "beanstalk", "amplify", "rds", "aurora", "redshift"
            ],
            "tools_methodologies": [
                "git", "github", "gitlab", "bitbucket", "ci/cd", "jenkins", "travis", "github actions",
                "circleci", "jira", "confluence", "agile", "scrum", "kanban", "tdd", "bdd",
                "devops", "figma", "sketch", "adobe xd", "photoshop", "illustrator", "invision",
                "zeplin", "selenium", "cypress", "jest", "mocha", "pytest", "junit", "gradle", "maven"
            ],
            "ml_ai": [
                "machine learning", "artificial intelligence", "ai", "ml", "deep learning", "neural networks",
                "nlp", "natural language processing", "computer vision", "cv", "reinforcement learning",
                "supervised learning", "unsupervised learning", "classification", "regression", "clustering",
                "transformers", "gpt", "bert", "llm", "large language model", "generative ai", "genai",
                "data science", "big data", "data analytics", "data engineering", "data mining"
            ],
            "mobile": [
                "ios", "android", "react native", "flutter", "swift", "kotlin", "objective-c",
                "mobile development", "app development", "pwa", "progressive web app",
                "xamarin", "cordova", "ionic", "nativescript", "swiftui", "jetpack compose"
            ],
            "architecture_patterns": [
                "microservices", "rest", "restful", "graphql", "soap", "mvc", "mvvm", "mvp",
                "serverless", "event-driven", "api gateway", "soa", "service oriented",
                "distributed systems", "cqrs", "domain driven design", "ddd", "event sourcing",
                "monolith", "microfrontend", "grpc", "websocket"
            ]
        }
    
    def _flatten_tech_skills(self) -> Set[str]:
        """Flatten all skills into a single set"""
        all_skills = set()
        for category in self.tech_skills.values():
            all_skills.update(category)
        return all_skills
    
    def _extract_skills_api(self, text: str) -> List[str]:
        """Extract skills using Hugging Face NER API"""
        if not self.api_key:
            logger.error("API key is required for Hugging Face inference")
            return []
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json={"inputs": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    break
                    
                logger.warning(f"API request failed (attempt {attempt+1}/{max_retries}): {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return []
        
        # Process response
        if response.status_code != 200:
            logger.error(f"API Error: Status code {response.status_code}")
            logger.debug(f"Response: {response.text}")
            return []
            
        # Parse results
        try:
            result = response.json()
            logger.debug(f"API Response: {result}")
            
            # Extract entity spans from response
            entities = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and 'entity_group' in item:
                        if item['entity_group'] in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
                            entities.append(item['word'].lower())
            
            return self._match_entities_to_skills(entities)
            
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return []
    
    def _extract_skills_spacy(self, text: str) -> List[str]:
        """Extract skills using local spaCy NER model"""
        try:
            doc = self.nlp(text)
            
            # Extract organizations and products as potential skills
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT']:
                    entities.append(ent.text.lower())
            
            # Use custom skill pattern matching
            tech_patterns = self._create_skill_patterns()
            matches = self._find_skill_patterns(text, tech_patterns)
            
            # Combine both approaches
            all_potential_skills = set(entities) | set(matches)
            return self._match_entities_to_skills(all_potential_skills)
            
        except Exception as e:
            logger.error(f"Error in spaCy processing: {str(e)}")
            return []
    
    def _create_skill_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Create regex patterns for tech skills"""
        patterns = []
        all_skills = self._flatten_tech_skills()
        
        # Create patterns for exact and partial matches
        for skill in all_skills:
            # Exact word boundary match
            pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
            patterns.append((skill, pattern))
            
            # Handle skills with periods like "node.js"
            if '.' in skill:
                clean_skill = skill.replace('.', r'\.')
                pattern = re.compile(r'\b' + clean_skill + r'\b', re.IGNORECASE)
                patterns.append((skill, pattern))
                
        return patterns
    
    def _find_skill_patterns(self, text: str, patterns: List[Tuple[str, re.Pattern]]) -> Set[str]:
        """Find all skill pattern matches in text"""
        matches = set()
        for skill, pattern in patterns:
            if pattern.search(text):
                matches.add(skill)
        return matches
    
    def _match_entities_to_skills(self, entities: List[str]) -> List[str]:
        """Match extracted entities to known tech skills"""
        matched_skills = set()
        all_skills = self._flatten_tech_skills()
        
        # Direct matches
        for entity in entities:
            entity = entity.lower()
            if entity in all_skills:
                matched_skills.add(entity)
                continue
                
            # Check for partial matches
            for skill in all_skills:
                if (skill in entity or entity in skill) and len(entity) > 2:
                    matched_skills.add(skill)
        
        # Add category-specific patterns
        text = " ".join(entities).lower()
        if 'ml' in text or 'ai' in text or 'machine' in text or 'learning' in text:
            matched_skills.update(['machine learning', 'ai'])
            
        if 'cloud' in text or 'aws' in text or 'azure' in text:
            matched_skills.update(['cloud computing'])
        
        return list(matched_skills)
    
    def _augment_with_related_skills(self, found_skills: List[str]) -> List[str]:
        """Augment found skills with related technologies"""
        augmented_skills = set(found_skills)
        
        # Dictionary of related skills
        related_skills = {
            "python": ["django", "flask", "pandas", "numpy", "pytorch", "tensorflow"],
            "javascript": ["react", "vue", "angular", "node.js", "express"],
            "java": ["spring", "spring boot", "hibernate", "maven"],
            "cloud": ["aws", "azure", "gcp", "kubernetes", "docker"],
            "aws": ["ec2", "s3", "lambda", "dynamodb"],
            "devops": ["docker", "kubernetes", "jenkins", "terraform", "ci/cd"],
            "react": ["javascript", "redux", "react native", "jsx"],
            "machine learning": ["python", "tensorflow", "pytorch", "scikit-learn"],
            "data science": ["python", "r", "pandas", "numpy", "sql"]
        }
        
        # Add related skills based on found skills
        for skill in found_skills:
            if skill in related_skills:
                # Add related skills with a limit to avoid over-expansion
                for related in related_skills[skill][:3]:  # Limit to 3 related skills
                    augmented_skills.add(related)
        
        return list(augmented_skills)
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize extracted skills"""
        categorized = {}
        
        for category, category_skills in self.tech_skills.items():
            category_matches = []
            for skill in skills:
                if skill in category_skills:
                    category_matches.append(skill)
            
            if category_matches:
                categorized[category] = category_matches
                
        return categorized
    
    def extract_skills_from_resume(self, resume_path: str, augment: bool = True) -> Dict[str, Any]:
        """
        Extract technical skills from resume
        
        Args:
            resume_path: Path to resume file
            augment: Whether to augment with related skills
            
        Returns:
            Dictionary with extracted skills and metadata
        """
        try:
            # Read resume content
            resume_text = Path(resume_path).read_text(encoding='utf-8')
            logger.info(f"Read resume from {resume_path}, length: {len(resume_text)} characters")
            
            # Extract skills based on method
            if self.use_api and self.api_key:
                raw_skills = self._extract_skills_api(resume_text)
            else:
                raw_skills = self._extract_skills_spacy(resume_text)
                
            # Augment with related skills if requested
            if augment and raw_skills:
                augmented_skills = self._augment_with_related_skills(raw_skills)
                logger.info(f"Augmented {len(raw_skills)} skills to {len(augmented_skills)} skills")
            else:
                augmented_skills = raw_skills
            
            # Categorize skills
            categorized_skills = self._categorize_skills(augmented_skills)
            
            # Prepare result
            result = {
                "extracted_skills": raw_skills,
                "augmented_skills": augmented_skills,
                "categorized_skills": categorized_skills,
                "skill_count": len(augmented_skills),
                "extraction_method": "api" if self.use_api else "spacy"
            }
            
            logger.info(f"Successfully extracted {result['skill_count']} skills from resume")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return {
                "extracted_skills": [],
                "augmented_skills": [],
                "categorized_skills": {},
                "skill_count": 0,
                "error": str(e)
            }

# Function to scrape job postings using Playwright
def scrape_jobs(company_urls, keywords, match_threshold=35):
    job_listings = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Use Chromium in headless mode
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        page = context.new_page()
        page.set_default_timeout(30000)  # 30 seconds

        for company in company_urls:
            url = company.get("Website", "")
            company_name = company.get("Company", "Unknown")
            
            if not url or not url.startswith(('http://', 'https://')):
                logger.warning(f"Skipping invalid URL for {company_name}: {url}")
                continue

            try:
                # Open the webpage using Playwright
                logger.info(f"Scraping {company_name}: {url}...")
                response = page.goto(url, wait_until="domcontentloaded", timeout=20000)
                
                if not response or response.status >= 400:
                    logger.warning(f"Error response {response.status if response else 'none'} from {url}")
                    continue
                
                # Wait for content to load
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    # Continue even if waiting for networkidle times out
                    pass
                
                # Get the page content
                page_content = page.content().lower()  # Get the page content

                # Count matched keywords in job description
                matched_keywords = []
                for keyword in keywords:
                    if keyword.lower() in page_content:
                        matched_keywords.append(keyword)
                
                matched_count = len(matched_keywords)
                # Calculate percentage of matched keywords
                matched_percentage = (matched_count / len(keywords)) * 100

                # If at least 50% of keywords match, save the job listing
                if matched_percentage >= match_threshold:
                    job_listing = {
                        "Company": company_name,
                        "URL": url,
                        "Matched Percentage": round(matched_percentage, 2),
                        "Matched Keywords": ", ".join(matched_keywords),
                        "Keyword Count": matched_count
                    }
                    
                    # Try to extract job title if available
                    try:
                        job_title = page.evaluate("""
                            () => {
                                const titleElements = Array.from(document.querySelectorAll('h1, h2')).filter(el => 
                                    el.innerText.toLowerCase().includes('job') || 
                                    el.innerText.toLowerCase().includes('career') ||
                                    el.innerText.toLowerCase().includes('position')
                                );
                                return titleElements.length > 0 ? titleElements[0].innerText.trim() : '';
                            }
                        """)
                        if job_title:
                            job_listing["Possible Job Title"] = job_title
                    except Exception:
                        pass  # Ignore errors in job title extraction
                    
                    job_listings.append(job_listing)
                    logger.info(f"Job listing at {url} matches {matched_percentage:.2f}% of your resume keywords.")

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")

            time.sleep(1)  # Prevent being blocked

        context.close()
        browser.close()  # Close the browser

    return job_listings

class JobScraper:
    def __init__(self, keywords, max_workers=5, match_threshold=35):
        self.keywords = keywords
        self.max_workers = max_workers
        self.match_threshold = match_threshold
        self.job_listings = []
        self.lock = Lock()
        self.progress_lock = Lock()
        self.processed_companies = 0
        self.total_companies = 0
        self.browser_pool = []
        self.start_time = time.time()
        
    def _update_progress(self):
        """Update and display progress information"""
        with self.progress_lock:
            self.processed_companies += 1
            progress = (self.processed_companies / self.total_companies) * 100
            elapsed = time.time() - self.start_time
            companies_per_second = self.processed_companies / elapsed if elapsed > 0 else 0
            estimated_total = (self.total_companies / companies_per_second) if companies_per_second > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            
            logger.info(
                f"Progress: {progress:.1f}% ({self.processed_companies}/{self.total_companies}) | "
                f"Speed: {companies_per_second:.2f}/sec | "
                f"Est. remaining: {remaining/60:.1f} min"
            )

    def initialize_browser_pool(self, count: int = None):
        """Initialize a pool of browser instances"""
        if count is None:
            count = self.max_workers
            
        logger.info(f"Initializing {count} browser instances")
        playwright = sync_playwright().start()
        
        for _ in range(count):
            browser = playwright.chromium.launch(headless=True)
            self.browser_pool.append((playwright, browser))
            
        logger.info(f"Browser pool initialized with {len(self.browser_pool)} instances")
        return playwright

    def _close_browser_pool(self):
        """Close all browser instances in the pool"""
        for playwright, browser in self.browser_pool:
            try:
                browser.close()
                playwright.stop()
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
                
        logger.info("Browser pool closed")

    def scrape_company(self, company):
        try:
            company_results = scrape_jobs([company], self.keywords, self.match_threshold)
            if company_results:
                with self.lock:
                    self.job_listings.extend(company_results)
            self._update_progress()
            return company_results
        except Exception as e:
            logger.error(f"Error scraping {company.get('Company', 'Unknown')}: {str(e)}")
            self._update_progress()
            return []

    def scrape_all_companies(self, companies):
        self.total_companies = len(companies)
        self.processed_companies = 0
        self.start_time = time.time()
        
        logger.info(f"Starting scraping of {self.total_companies} companies with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all companies to the thread pool
            future_to_company = {
                executor.submit(self.scrape_company, company): company 
                for company in companies
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    future.result()  # Results already added in scrape_company
                except Exception as e:
                    logger.error(f"Thread error for {company.get('Company', 'Unknown')}: {str(e)}")
                    
        logger.info(f"Scraping complete. Found {len(self.job_listings)} matching positions.")
        return self.job_listings

def main():
    """Main function to run the job scraper"""
    logger.info("Starting job search process...")
    start_time = time.time()
    
    # Configuration
    SHEET_URL = "https://docs.google.com/spreadsheets/d/1-Be2li67qv0FWs_U2_fXTLnhWiYjPvvDZ94cTeNxSkg/edit#gid=0"
    RESUME_PATH = "/Users/admin/Desktop/google/resume.txt"
    CREDENTIALS_PATH = "/Users/admin/Desktop/google/arctic-acolyte-451122-s2-e8fc9d28a3fa.json"
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_TOKEN")
    MAX_WORKERS = 8
    MATCH_THRESHOLD = 30
    OUTPUT_FILE = "filtered_job_listings.csv"
    USE_NER_EXTRACTOR = True  # Set to True to use the new NER skill extractor
    
    # 1. Extract skills from resume
    logger.info("Extracting skills from resume...")
    if USE_NER_EXTRACTOR:
        # Use the new NER-based skill extractor
        skill_extractor = NERSkillExtractor(use_api=True, api_key=HUGGINGFACE_API_KEY)
        skills_result = skill_extractor.extract_skills_from_resume(RESUME_PATH)
        keywords = skills_result['augmented_skills']
        
        # Save extracted skills to file for reference
        with open("extracted_skills.json", "w") as f:
            json.dump(skills_result, f, indent=2)
        
        logger.info(f"Extracted {skills_result['skill_count']} skills using NER")
        if skills_result['categorized_skills']:
            for category, skills in skills_result['categorized_skills'].items():
                logger.info(f"{category}: {', '.join(skills)}")
    else:
        # Fallback to hardcoded keywords if NER extraction is disabled
        keywords = ["python", "java", "javascript", "cloud", "aws", "api"]
        logger.info(f"Using fallback keywords: {keywords}")
    
    if not keywords:
        logger.error("No keywords extracted or provided. Cannot continue.")
        return
        
    logger.info(f"Using {len(keywords)} keywords for job matching")
    
    # 2. Fetch companies from Google Sheet
    logger.info("Fetching companies from Google Sheet...")
    companies = get_google_sheet(SHEET_URL, CREDENTIALS_PATH)
    
    if not companies:
        logger.error("No companies found in sheet. Please check your credentials and sheet URL.")
        return
        
    logger.info(f"Found {len(companies)} companies in the sheet")
    
    # 3. Scrape job listings
    logger.info(f"Starting job scraping with {MAX_WORKERS} workers...")
    scraper = JobScraper(keywords, max_workers=MAX_WORKERS, match_threshold=MATCH_THRESHOLD)
    job_listings = scraper.scrape_all_companies(companies)
    
    # 4. Save results
    if job_listings:
        df = pd.DataFrame(job_listings)
        # Sort by match percentage (descending)
        df = df.sort_values(by='Matched Percentage', ascending=False)
        df.to_csv(OUTPUT_FILE, index=False)
        
        # Generate summary statistics
        total_jobs = len(job_listings)
        avg_match = df['Matched Percentage'].mean()
        high_matches = len(df[df['Matched Percentage'] >= 60])
        
        logger.info(f"Summary: Found {total_jobs} matching positions")
        logger.info(f"Average match percentage: {avg_match:.1f}%")
        logger.info(f"High-quality matches (≥60%): {high_matches}")
        logger.info(f"Results saved to '{OUTPUT_FILE}'")
    else:
        logger.warning("No job listings found.")
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()