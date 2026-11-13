"""Headlines from established newsrooms and official US government sites.

Google News is only the index. Each outlet, and each agency, is its own
source: three newsrooms reporting a result are three sources, and the White
House and the FDA are two. Outlets outside a curated list of established
newsrooms are dropped, so a blog cannot outvote a wire service. Opinion
columns are dropped too: they argue, they do not report.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # types only; parsing goes through defusedxml
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import search_terms
from verifai.retrieval.types import Passage, RetrievalResult

_FEED = "https://news.google.com/rss/search"
_RECENT_DAYS = 30

# Wire services, national papers and broadcasters, and the main international
# newsrooms, each with a record of corrections. Subdomains count as the outlet.
MAJOR_OUTLETS: dict[str, str] = {
    # Wire services
    "apnews.com": "AP News",
    "reuters.com": "Reuters",
    "afp.com": "AFP",
    # United States
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "wsj.com": "The Wall Street Journal",
    "usatoday.com": "USA Today",
    "latimes.com": "Los Angeles Times",
    "bostonglobe.com": "The Boston Globe",
    "chicagotribune.com": "Chicago Tribune",
    "npr.org": "NPR",
    "pbs.org": "PBS",
    "cbsnews.com": "CBS News",
    "nbcnews.com": "NBC News",
    "abcnews.go.com": "ABC News",
    "cnn.com": "CNN",
    "foxnews.com": "Fox News",
    "bloomberg.com": "Bloomberg",
    "cnbc.com": "CNBC",
    "politico.com": "Politico",
    "axios.com": "Axios",
    "thehill.com": "The Hill",
    "time.com": "Time",
    "newsweek.com": "Newsweek",
    "propublica.org": "ProPublica",
    "c-span.org": "C-SPAN",
    # United Kingdom and Europe
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "theguardian.com": "The Guardian",
    "ft.com": "Financial Times",
    "economist.com": "The Economist",
    "thetimes.com": "The Times",
    "telegraph.co.uk": "The Telegraph",
    "independent.co.uk": "The Independent",
    "sky.com": "Sky News",
    "dw.com": "DW",
    "france24.com": "France 24",
    "euronews.com": "Euronews",
    "lemonde.fr": "Le Monde",
    "spiegel.de": "Der Spiegel",
    "elpais.com": "El País",
    # Elsewhere
    "aljazeera.com": "Al Jazeera",
    "cbc.ca": "CBC News",
    "theglobeandmail.com": "The Globe and Mail",
    "abc.net.au": "ABC News (Australia)",
    "smh.com.au": "The Sydney Morning Herald",
    "nhk.or.jp": "NHK",
    "japantimes.co.jp": "The Japan Times",
    "scmp.com": "South China Morning Post",
    "thehindu.com": "The Hindu",
    "hindustantimes.com": "Hindustan Times",
    "indianexpress.com": "The Indian Express",
    "straitstimes.com": "The Straits Times",
    "timesofisrael.com": "The Times of Israel",
    "premiumtimesng.com": "Premium Times",
    "punchng.com": "Punch",
    "channelstv.com": "Channels Television",
    "nation.africa": "Nation",
    "news24.com": "News24",
    "dailymaverick.co.za": "Daily Maverick",
}

# Official US federal sites. An agency's own statement is evidence of what
# the government said or did; it is still one source, as any outlet is.
GOVERNMENT_SITES: dict[str, str] = {
    "whitehouse.gov": "The White House",
    "fda.gov": "U.S. Food and Drug Administration",
    "cdc.gov": "Centers for Disease Control and Prevention",
    "nih.gov": "National Institutes of Health",
    "hhs.gov": "U.S. Department of Health and Human Services",
    "cms.gov": "Centers for Medicare & Medicaid Services",
    "congress.gov": "Congress.gov",
    "senate.gov": "U.S. Senate",
    "house.gov": "U.S. House of Representatives",
    "supremecourt.gov": "Supreme Court of the United States",
    "justice.gov": "U.S. Department of Justice",
    "state.gov": "U.S. Department of State",
    "defense.gov": "U.S. Department of Defense",
    "war.gov": "U.S. Department of War",
    "treasury.gov": "U.S. Department of the Treasury",
    "usda.gov": "U.S. Department of Agriculture",
    "epa.gov": "U.S. Environmental Protection Agency",
    "energy.gov": "U.S. Department of Energy",
    "ed.gov": "U.S. Department of Education",
    "dhs.gov": "U.S. Department of Homeland Security",
    "dol.gov": "U.S. Department of Labor",
    "bls.gov": "U.S. Bureau of Labor Statistics",
    "census.gov": "U.S. Census Bureau",
    "federalreserve.gov": "Federal Reserve",
    "sec.gov": "U.S. Securities and Exchange Commission",
    "ftc.gov": "Federal Trade Commission",
    "fcc.gov": "Federal Communications Commission",
    "nasa.gov": "NASA",
    "noaa.gov": "NOAA",
    "usgs.gov": "U.S. Geological Survey",
}

# Google News accepts a long OR of site: filters. These are the agencies a
# news or health claim most often names; any .gov result is still kept.
_GOVERNMENT_QUERY_SITES = (
    "whitehouse.gov",
    "fda.gov",
    "cdc.gov",
    "nih.gov",
    "hhs.gov",
    "congress.gov",
    "senate.gov",
    "house.gov",
    "justice.gov",
    "state.gov",
    "defense.gov",
    "war.gov",
    "treasury.gov",
    "usda.gov",
    "epa.gov",
    "supremecourt.gov",
)

_OPINION = re.compile(
    r"^\s*(?:opinion|op-ed|editorial|column|commentary|analysis|letters?)\s*[|:–—-]"
    r"|\|\s*opinion\b|\bopinion\s*\|",
    re.IGNORECASE,
)


class NewsSearchRetriever:
    """Recent and archived headlines from a fixed set of trusted publishers."""

    def __init__(
        self,
        http: PoliteHttp,
        *,
        name: str,
        publishers: dict[str, str],
        sites: tuple[str, ...] = (),
        family_prefix: str = "outlet",
        any_gov: bool = False,
        limit: int = 5,
    ) -> None:
        self.name = name
        self.http = http
        self.publishers = publishers
        self.sites = sites
        self.family_prefix = family_prefix
        self.any_gov = any_gov
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        words = search_terms(query, limit=10)
        if len(words) < 2:
            return RetrievalResult(self.name)
        search = " ".join(words)
        if self.sites:
            search += " (" + " OR ".join(f"site:{site}" for site in self.sites) + ")"
        # The last month's reporting first, so a result is judged on what is
        # true now. Older reporting fills in for settled history ("the FDA
        # approved the first COVID-19 vaccine in 2021").
        passages: list[Passage] = []
        seen: set[str] = set()
        errors: list[str] = []
        for window in (f"{search} when:{_RECENT_DAYS}d", search):
            items, error = self._search(window)
            if error:
                errors.append(error)
                continue
            for item in items:
                passage = self._passage(item)
                if passage is None or passage.family in seen:
                    continue
                seen.add(passage.family)
                passages.append(passage)
            if len(passages) >= self.limit:
                break
        if len(errors) == 2:
            return RetrievalResult(self.name, unavailable=True, error=errors[0])
        passages = passages[: self.limit]
        passages.sort(key=lambda passage: passage.published or date.min, reverse=True)
        return RetrievalResult(self.name, passages=passages)

    def _search(self, search: str) -> tuple[list[ET.Element], str | None]:
        result = self.http.get(
            _FEED,
            params={"q": search, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={"Accept": "application/rss+xml, application/xml"},
        )
        if result.error or result.status_code != 200:
            return [], result.error or f"Google News returned {result.status_code}"
        try:
            root = SafeET.fromstring(result.content)
        except (ET.ParseError, DefusedXmlException):
            return [], "the news feed was not XML"
        return list(root.iter("item")), None

    def _passage(self, item: ET.Element) -> Passage | None:
        source = item.find("source")
        home = source.get("url", "") if source is not None else ""
        domain = _domain(home)
        publisher = self._publisher(domain)
        if publisher is None:
            return None
        outlet_name, outlet_domain = publisher
        title = _headline(
            item.findtext("title") or "", (source.text or "") if source is not None else ""
        )
        if len(title) < 12 or _OPINION.search(title):
            return None
        url = (item.findtext("link") or "").strip() or None
        published = _published(item.findtext("pubDate"))
        when = f", published {published:%-d %B %Y}{_age(published)}" if published else ""
        return Passage(
            source=self.name,
            family=f"{self.family_prefix}:{outlet_domain}",
            title=title,
            url=url,
            text=f"{title}\n(Headline from {outlet_name}{when}.)",
            year=published.year if published else None,
            published=published,
            venue=outlet_name,
            speaker=outlet_name,
        )

    def _publisher(self, domain: str) -> tuple[str, str] | None:
        """The publisher's name and the domain it is counted under."""
        for known, label in self.publishers.items():
            if domain == known or domain.endswith(f".{known}"):
                return label, known
        if self.any_gov and domain.endswith((".gov", ".mil")):
            return domain, domain
        return None


def _domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def _headline(title: str, publisher: str) -> str:
    """Google appends " - Outlet name" to each headline."""
    title = " ".join(title.split())
    publisher = " ".join((publisher or "").split())
    if publisher and title.endswith(f" - {publisher}"):
        title = title[: -len(publisher) - 3]
    return title.strip()


def _age(published: date) -> str:
    """How old a report is, so a judge reads an old headline as old news."""
    days = (datetime.now(UTC).date() - published).days
    if days < 45:
        return ""
    months = round(days / 30.4)
    span = f"{months} months" if months < 24 else f"{round(days / 365.25)} years"
    return f", {span} ago: it shows what was true then, not now"


def _published(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def build_major_news(http: PoliteHttp) -> NewsSearchRetriever:
    return NewsSearchRetriever(http, name="major_news", publishers=MAJOR_OUTLETS, limit=5)


def build_us_government(http: PoliteHttp) -> NewsSearchRetriever:
    return NewsSearchRetriever(
        http,
        name="us_government",
        publishers=GOVERNMENT_SITES,
        sites=_GOVERNMENT_QUERY_SITES,
        family_prefix="gov",
        any_gov=True,
        limit=3,
    )
