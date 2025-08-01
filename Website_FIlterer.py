import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.parser import isoparse
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components
import re
from io import StringIO
import os
# Custom CSSF
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

components.html(
    """
        <script>
            (function () {
    const threshold = 160;

    // Check for DevTools every second
    setInterval(() => {
      const devtoolsOpen = (
        window.outerWidth - window.innerWidth > threshold ||
        window.outerHeight - window.innerHeight > threshold
      );
      if (devtoolsOpen) {
        window.location.replace("https://seekgps.com");
      }
    }, 1000);

    // Block Ctrl+S and Ctrl+U
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 's' || e.key.toLowerCase() === 'u')) {
        e.preventDefault();
        alert('This action is disabled on this page.');
        return false;
      }
    });
  })();
        </script>
    """,
    height=0,
)
# Main category keywords
MAIN_CATEGORIES = {
    'blog': ['blog', 'post', 'article', 'news', 'journal', 'writing', 'editorial', 'author'],
    'ecommerce': ['shop', 'store', 'cart', 'buy', 'purchase', 'checkout', 'product', 'price', 'order', 'shipping'],
    'service': ['service', 'consulting', 'solution', 'support', 'help', 'contact', 'about', 'professional', 'agency']
}

# Niche category keywords
NICHE_CATEGORIES = {
    'technology': ['tech', 'technology', 'software', 'hardware', 'programming', 'developer', 'code'],
    'fashion': ['fashion', 'style', 'clothing', 'dress', 'apparel', 'outfit', 'beauty'],
    'health': ['health', 'medical', 'wellness', 'fitness', 'doctor', 'hospital', 'diet'],
    'finance': ['finance', 'money', 'banking', 'invest', 'stock', 'market', 'trading'],
    'education': ['education', 'school', 'university', 'college', 'learn', 'student', 'teach'],
    'travel': ['travel', 'tourism', 'vacation', 'destination', 'hotel', 'flight', 'tour'],
    'food': ['food', 'restaurant', 'recipe', 'cooking', 'dining', 'cuisine', 'meal'],
    'sports': ['sport', 'football', 'basketball', 'tennis', 'athletics', 'golf', 'fitness'],
    'entertainment': ['entertainment', 'movie', 'music', 'game', 'celebrity', 'show', 'film'],
    'automotive': ['car', 'vehicle', 'auto', 'automotive', 'parts', 'garage', 'mechanic'],
    'realestate': ['real estate', 'property', 'housing', 'mortgage', 'rental', 'apartment'],
    'business': ['business', 'company', 'corporate', 'enterprise', 'startup', 'entrepreneur']
}

# Language code mapping (common ones)
LANGUAGE_MAP = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi'
}

# TLD categories
TLD_CATEGORIES = {
    'gTLD': ['com', 'net', 'org', 'info', 'biz', 'name', 'pro', 'xyz', 'online', 'site', 'tech', 'store', 'shop'],
    'ccTLD': ['us', 'uk', 'ca', 'au', 'de', 'fr', 'it', 'es', 'in', 'jp', 'cn', 'br', 'ru', 'mx', 'ae'],
    'New gTLD': ['app', 'blog', 'cloud', 'dev', 'ai', 'io', 'ly', 'me', 'tv', 'fm', 'to', 'co', 'academy', 'agency'],
    'Infrastructure': ['arpa', 'example', 'invalid', 'localhost', 'test'],
    'Geographic': ['nyc', 'london', 'paris', 'tokyo', 'berlin', 'moscow', 'dubai', 'sydney'],
    'Brand': ['google', 'amazon', 'apple', 'microsoft', 'facebook', 'bmw', 'audi', 'nike'],
    'Other': ['gov', 'edu', 'mil', 'int', 'post', 'tel', 'mobi', 'asia', 'jobs']
}

def get_tld_category(tld):
    """Categorize TLDs"""
    tld = tld.lower()
    for category, tlds in TLD_CATEGORIES.items():
        if tld in tlds:
            return category
    return 'Other'

def extract_domain_info(url):
    """Extract domain, TLD, and domain age"""
    try:
        # Remove protocol and path
        domain = re.sub(r'^https?://(www\.)?', '', url)
        domain = domain.split('/')[0]
        
        # Extract TLD
        tld = domain.split('.')[-1]
        
        return {
            'domain': domain,
            'tld': tld,
            'tld_category': get_tld_category(tld)
        }
    except:
        return {
            'domain': url,
            'tld': 'unknown',
            'tld_category': 'Unknown'
        }

def get_custom_categories():
    """Get custom niche categories from user input using Streamlit"""
    st.markdown("""
    <div class="custom-category-box">
        <h3>Custom Niche Category Setup</h3>
        <p>Add your own niche categories with keywords.</p>
        <p><strong>Format:</strong> category_name:keyword1,keyword2,keyword3</p>
        <p><strong>Example:</strong> photography:camera,lens,photo,photographer</p>
    </div>
    """, unsafe_allow_html=True)
    
    custom_categories = {}
    
    with st.form("custom_categories_form"):
        cols = st.columns(2)
        category = cols[0].text_input("Category Name").strip().lower()
        keywords = cols[1].text_input("Keywords (comma separated)").strip().lower()
        submitted = st.form_submit_button("Add Category", use_container_width=True)
        
        if submitted and category and keywords:
            keyword_list = [k.strip() for k in keywords.split(',')]
            custom_categories[category] = keyword_list
            st.success(f"✅ Added niche: {category} with {len(keyword_list)} keywords")
    
    return custom_categories

def merge_keywords(default, custom):
    """Merge default and custom keywords"""
    merged = default.copy()
    for category, keywords in custom.items():
        if category in merged:
            merged[category].extend(keywords)
            merged[category] = list(set(merged[category]))  # Remove duplicates
        else:
            merged[category] = keywords
    return merged

def get_site_content(url, timeout=15, retries=3):
    """Fetch the HTML content of the website with retries and timeout"""
    for attempt in range(retries):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text, None, response.url  # Return final URL (after redirects)
        except Exception as e:
            if attempt == retries - 1:
                return None, str(e), url
            time.sleep(2)  # Wait 2 seconds before retrying

def detect_language(soup):
    """Detect language from HTML lang attribute"""
    try:
        lang = soup.html.get('lang', '').split('-')[0].lower()  # Get primary language code
        return LANGUAGE_MAP.get(lang, lang)
    except:
        return "Unknown"

def analyze_content(content, main_categories, niche_categories):
    """Analyze content and return main type, niches, and language"""
    if not content:
        return "Unknown", [], "Unknown", "No content"

    try:
        soup = BeautifulSoup(content, 'html.parser')
        content_lower = content.lower()
        
        # Detect language
        language = detect_language(soup)
        
        # Analyze main category with weighted scores
        main_scores = {category: 0 for category in main_categories}
        
        # Weight certain keywords higher for ecommerce
        ecommerce_weighted = ['cart', 'checkout', 'price', 'product', 'shop', 'store']
        
        for category, keywords in main_categories.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Give higher weight to ecommerce-specific keywords
                    if category == 'ecommerce' and keyword in ecommerce_weighted:
                        main_scores[category] += 3
                    else:
                        main_scores[category] += 1
        
        # Get the category with highest score
        if not any(main_scores.values()):
            main_type = "Unknown"
        else:
            # Get the category with maximum score
            max_score = max(main_scores.values())
            # If there's a tie, prioritize ecommerce > service > blog
            if list(main_scores.values()).count(max_score) > 1:
                if main_scores['ecommerce'] == max_score:
                    main_type = 'ecommerce'
                elif main_scores['service'] == max_score:
                    main_type = 'service'
                else:
                    main_type = 'blog'
            else:
                main_type = max(main_scores.items(), key=lambda x: x[1])[0]
        
        # Analyze niche categories
        niche_matches = []
        for niche, keywords in niche_categories.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches >= 2:  # Minimum 2 keyword matches for a niche
                niche_matches.append(niche)
        
        return main_type, niche_matches, language, None
        
    except Exception as e:
        return "Unknown", [], "Unknown", str(e)

def detect_articles(content):
    """Detect articles in the content"""
    if not content:
        return []

    try:
        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        # Look for articles in common tags
        for tag in soup.find_all(['article', 'div', 'section']):
            classes = tag.get('class', [])
            class_str = ' '.join(classes).lower() if classes else ''
            
            # Common article indicators
            is_article = (
                tag.name == 'article' or
                'post' in class_str or
                'article' in class_str or
                'blog' in class_str or
                'story' in class_str
            )
            
            if is_article:
                try:
                    title = tag.find(['h1', 'h2', 'h3']).text if tag.find(['h1', 'h2', 'h3']) else 'No Title'
                    date_tag = tag.find('time')
                    date = date_tag['datetime'] if date_tag and 'datetime' in date_tag.attrs else 'Unknown Date'
                    articles.append((title, date))
                except:
                    continue

        return articles
    except Exception as e:
        return []

def is_recent(article_date):
    """Check if the article is recent"""
    try:
        if article_date == 'Unknown Date':
            return False
        published_date = isoparse(article_date)
        now = datetime.now(published_date.tzinfo)
        return (now - published_date).days <= 30
    except:
        return False

def classify_website(url, main_categories, niche_categories):
    """Classify a website with comprehensive details"""
    try:
        logger.info(f"Processing website: {url}")
        content, error, final_url = get_site_content(url)
        if error:
            return {
                "url": final_url,
                "main_type": "Error",
                "niches": [],
                "language": "Unknown",
                "recent_articles": [],
                "success": False,
                "error": error
            }

        main_type, niches, language, cat_error = analyze_content(content, main_categories, niche_categories)
        articles = detect_articles(content) if content else []
        recent_articles = [(title, date) for title, date in articles if is_recent(date)]
        
        # Extract domain info
        domain_info = extract_domain_info(final_url)

        return {
            "url": final_url,
            "domain": domain_info['domain'],
            "tld": domain_info['tld'],
            "tld_category": domain_info['tld_category'],
            "main_type": main_type,
            "niches": niches,
            "language": language,
            "recent_articles": recent_articles,
            "success": True,
            "error": cat_error
        }
    except Exception as e:
        return {
            "url": url,
            "domain": url,
            "tld": "unknown",
            "tld_category": "Unknown",
            "main_type": "Error",
            "niches": [],
            "language": "Unknown",
            "recent_articles": [],
            "success": False,
            "error": str(e)
        }

def process_websites(df, main_categories, niche_categories, progress_bar):
    """Process websites while maintaining original data alignment"""
    results = []
    total = len(df)
    
    # Create a dictionary to store results with original index
    results_dict = {index: None for index in range(len(df))}
    
    # Add progress tracking variables
    start_time = time.time()
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                classify_website, 
                row['Website'], 
                main_categories, 
                niche_categories
            ): index 
            for index, row in df.iterrows()
        }

        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                results_dict[index] = {
                    "Domain": result['domain'],
                    "TLD": result['tld'],
                    "TLD Category": result['tld_category'],
                    "Main Type": result['main_type'],
                    "Niches": ", ".join(result['niches']) if result['niches'] else "None",
                    "Language": result['language'],
                    "Recent Articles": len(result['recent_articles']),
                    "Multiple Niches": len(result['niches']) > 1,
                    "Multiple Main Types": "," in result['main_type'],
                    "Success": result['success'],
                    "Error": result['error']
                }
            except Exception as e:
                results_dict[index] = {
                    "Domain": result.get('domain', 'Unknown'),
                    "TLD": result.get('tld', 'unknown'),
                    "TLD Category": result.get('tld_category', 'Unknown'),
                    "Main Type": "Error",
                    "Niches": "None",
                    "Language": "Unknown",
                    "Recent Articles": 0,
                    "Multiple Niches": False,
                    "Multiple Main Types": False,
                    "Success": False,
                    "Error": str(e)
                }
            
            # Update progress tracking
            processed_count += 1
            progress_percent = int((processed_count / total) * 100)
            
            # Calculate estimated time remaining
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / processed_count
            remaining_items = total - processed_count
            estimated_remaining = avg_time_per_item * remaining_items
            
            # Format time display
            if estimated_remaining > 60:
                time_display = f"{estimated_remaining/60:.1f} minutes remaining"
            else:
                time_display = f"{estimated_remaining:.0f} seconds remaining"
            
            # Update progress bar with additional info
            progress_bar.progress(progress_percent/100, text=f"Processing {processed_count}/{total} websites ({progress_percent}%) - {time_display}")
    
    # Convert results dictionary to DataFrame in correct order
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    
    # Join with original DataFrame while preserving order
    final_df = df.join(results_df)
    
    return final_df

def filter_data(df, filters):
    """Filter data based on user selections"""
    filtered_df = df.copy()
    
    if filters['main_type'] != 'All':
        filtered_df = filtered_df[filtered_df['Main Type'].str.contains(filters['main_type'], case=False, na=False)]
    
    if filters['niche'] != 'All':
        filtered_df = filtered_df[filtered_df['Niches'].str.contains(filters['niche'], case=False, na=False)]
    
    if filters['language'] != 'All':
        filtered_df = filtered_df[filtered_df['Language'].str.contains(filters['language'], case=False, na=False)]
    
    if filters['tld_category'] != 'All':
        filtered_df = filtered_df[filtered_df['TLD Category'].str.contains(filters['tld_category'], case=False, na=False)]
    
    if filters['tld'] != 'All':
        filtered_df = filtered_df[filtered_df['TLD'].str.contains(filters['tld'], case=False, na=False)]
    
    if filters['success_only']:
        filtered_df = filtered_df[filtered_df['Success'] == True]
    
    return filtered_df

def main():
    st.set_page_config(
        initial_sidebar_state="collapsed"  # This will hide the sidebar
    )
    
    # Load custom CSS
    local_css("style.css")

    # Initialize session state for processed data and custom categories
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'custom_niches' not in st.session_state:
        st.session_state.custom_niches = {}
    if 'show_add_category' not in st.session_state:
        st.session_state.show_add_category = False

    # Add Category button - top right corner
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("➕ Add Category", key="add_category_btn"):
            st.session_state.show_add_category = not st.session_state.show_add_category

    # Show add category form if toggled
    if st.session_state.show_add_category:
        with st.expander("➕ Add Custom Category", expanded=True):
            with st.form("custom_categories_form"):
                cols = st.columns(2)
                category = cols[0].text_input("Category Name").strip().lower()
                keywords = cols[1].text_input("Keywords (comma separated)").strip().lower()
                submitted = st.form_submit_button("Add Category", use_container_width=True)
                
                if submitted and category and keywords:
                    keyword_list = [k.strip() for k in keywords.split(',')]
                    st.session_state.custom_niches[category] = keyword_list
                    st.success(f"✅ Added niche: {category} with {len(keyword_list)} keywords")
                    st.session_state.show_add_category = False  # Hide after adding

    # Merge default and custom categories
    niche_categories = merge_keywords(NICHE_CATEGORIES, st.session_state.custom_niches)

    # Show active categories info
    st.markdown(f"""
    <div class="active-categories">
        <p class="h3">Active Categories:</p>
        <p><strong>Main Types:</strong> {len(MAIN_CATEGORIES)}</p>
        <p><strong>Niche Categories:</strong> {len(niche_categories)} (including {len(st.session_state.custom_niches)} custom)</p>
        <p><strong>Sample Niches:</strong> {", ".join(list(niche_categories.keys())[:5])}...</p>
    </div>
    """, unsafe_allow_html=True)

    # File upload section
    st.markdown("""
    <div class="file-upload">
        <p class="h3">Upload Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file with websites", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Website' not in df.columns:
                    st.error("⚠️ CSV file must contain a 'Website' column with URLs")
                    st.stop()
                
                if st.button("Start Analysis", use_container_width=True, type="primary"):
                   with st.spinner("Processing websites..."):
                        progress_bar = st.progress(0, text="Preparing to process websites...")
                        result_df = process_websites(df, MAIN_CATEGORIES, niche_categories, progress_bar)
                        st.session_state.processed_data = result_df
                        progress_bar.empty()
                        st.success("✅  Analysis completed!")
                        st.markdown("""<hr>""", unsafe_allow_html=True)
                            # File upload section
                        st.markdown("""
                        <hr>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Main content area
    if st.session_state.processed_data is not None:
        result_df = st.session_state.processed_data
        
        # Display summary statistics
        st.markdown("""
        <div class="summary-section">
            <p class="h3">Analysis Summary:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        success_count = result_df['Success'].sum()
        error_count = len(result_df) - success_count
        
        col1.markdown("""
        <div class="metric-card">
            <p class="h2">Total Websites:</p>
            <p>{}</p>
        </div>
        """.format(len(result_df)), unsafe_allow_html=True)
        
        col2.markdown("""
        <div class="metric-card success">
            <p class="h2">Successful:</p>
            <p>{}</p>
        </div>
        """.format(success_count), unsafe_allow_html=True)
        
        col3.markdown("""
        <div class="metric-card error">
            <p class="h2">Errors:</p>
            <p>{}</p>
        </div>
        """.format(error_count), unsafe_allow_html=True)
        
        # Filters
        st.markdown("""
        <div class="filter-section">
            <p class="h1"></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get unique values for filters
        main_types = ['All'] + sorted(result_df['Main Type'].unique().tolist())
        niches = ['All'] + sorted(set([niche for niches in result_df['Niches'].str.split(', ') for niche in niches if niche != 'None']))
        languages = ['All'] + sorted(result_df['Language'].unique().tolist())
        tld_categories = ['All'] + sorted(result_df['TLD Category'].unique().tolist())
        tlds = ['All'] + sorted(result_df['TLD'].unique().tolist())
        
        # Create filter controls
        cols = st.columns(5)
        filters = {
            'main_type': cols[0].selectbox("Main Type", main_types),
            'niche': cols[1].selectbox("Niche", niches),
            'language': cols[2].selectbox("Language", languages),
            'tld_category': cols[3].selectbox("TLD Category", tld_categories),
            'tld': cols[4].selectbox("TLD", tlds),
            'success_only': st.checkbox("Show only successful classifications", True)
        }
        
        # Apply filters
        filtered_df = filter_data(result_df, filters)
        
        # Show filtered results
        st.markdown(f"""
        <div class="results-count">
            <p>Showing <strong>{len(filtered_df)}</strong> of <strong>{len(result_df)}</strong> websites</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download buttons
        st.markdown("""
        <div class="download-section">
            <p class="h1">Download Results :</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        if len(filtered_df) < len(result_df):
            with col1:
                if st.button("Download Filtered Data (CSV)", use_container_width=True):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Click to download",
                        data=csv,
                        file_name="filtered_websites.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col2:
            if st.button("Download All Data (CSV)", use_container_width=True):
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name="all_websites.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Detailed statistics
        st.markdown("""
        <div class="stats-section">
            <p class="h3">Detailed Statistics :</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(filtered_df) > 0:
            tab1, tab2, tab3, tab4 = st.tabs(["Main Types", "Niches", "Languages", "TLDs"])
            
            with tab1:
                st.markdown("### Main Type Distribution")
                st.bar_chart(filtered_df['Main Type'].value_counts())
            
            with tab2:
                st.markdown("### Niche Distribution")
                niche_counts = pd.Series([niche for niches in filtered_df['Niches'].str.split(', ') for niche in niches if niche != 'None'])
                st.bar_chart(niche_counts.value_counts())
            
            with tab3:
                st.markdown("### Language Distribution")
                st.bar_chart(filtered_df['Language'].value_counts())
            
            with tab4:
                st.markdown("### TLD Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Top 10 TLDs")
                    st.bar_chart(filtered_df['TLD'].value_counts().head(10))
                with col2:
                    st.markdown("#### TLD Categories")
                    st.bar_chart(filtered_df['TLD Category'].value_counts())
        else:
            st.warning("No data to display with current filters")

if __name__ == "__main__":
    main()
