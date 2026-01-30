"""
Web Scraper Module
Fetches and extracts text content from web pages.
"""

import requests
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urlparse
from typing import Optional


class WebScraper:
    """Scrapes web pages and extracts clean text content."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No wrapping

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL.

        Args:
            url: The URL to fetch

        Returns:
            Raw HTML content or None if fetch fails
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme in ['http', 'https']:
                raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
        except ValueError as e:
            print(f"Invalid URL: {e}")
            return None

    def extract_text(self, html_content: str) -> str:
        """
        Extract clean text from HTML content.

        Args:
            html_content: Raw HTML string

        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Try to find main content
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_='content') or
            soup.find('div', id='content') or
            soup.body or
            soup
        )

        # Convert to markdown-like text
        text = self.html_converter.handle(str(main_content))

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        text = '\n'.join(lines)

        return text

    def extract_metadata(self, html_content: str, url: str) -> dict:
        """
        Extract metadata from HTML content.

        Args:
            html_content: Raw HTML string
            url: The source URL

        Returns:
            Dictionary containing metadata
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title
        title = None
        if soup.title:
            title = soup.title.string
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content')

        # Extract description
        description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content')
        if not description:
            og_desc = soup.find('meta', property='og:description')
            if og_desc:
                description = og_desc.get('content')

        return {
            'url': url,
            'title': title or 'Unknown',
            'description': description or '',
            'domain': urlparse(url).netloc
        }

    def scrape(self, url: str) -> Optional[dict]:
        """
        Scrape a URL and return structured content.

        Args:
            url: The URL to scrape

        Returns:
            Dictionary with text content and metadata, or None if failed
        """
        html_content = self.fetch_url(url)
        if not html_content:
            return None

        text = self.extract_text(html_content)
        metadata = self.extract_metadata(html_content, url)

        return {
            'content': text,
            'metadata': metadata
        }