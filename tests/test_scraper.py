from datetime import date
from bs4 import BeautifulSoup
from unittest.mock import MagicMock
from texas_mushrooms.scraper import parse_index, parse_photos
from texas_mushrooms.models import DayPage


def test_parse_index_extracts_dates() -> None:
    # Minimal HTML snippet for indexc.htm
    html = """
    <html>
    <body>
        <a href="date-en/2023-01-01.htm">2023-01-01</a>
        <a href="date-en/2023-01-02.htm">2023-01-02</a>
        <a href="other.htm">Other</a>
    </body>
    </html>
    """

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.content = html.encode("utf-8")
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response

    # We need to mock get_soup or just let it use the mock session
    # Since parse_index calls get_soup which calls session.get, this works.

    days = parse_index(mock_session)

    assert len(days) == 2
    assert days[0].date == date(2023, 1, 2)  # Sorted descending
    assert str(days[0].url) == "https://www.texasmushrooms.org/date-en/2023-01-02.htm"
    assert days[1].date == date(2023, 1, 1)


def test_parse_photos_extracts_records() -> None:
    # Minimal HTML snippet for a day page
    html = """
    <html>
    <body>
        <p>
            <a href="/pictures/archives/test_img.jpg">
                <img src="thumb.jpg">
            </a>
            Found in the woods.
            <a href="fungi_en/agaricus_bisporus.htm">Agaricus bisporus</a>
        </p>
    </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    day = DayPage(
        date=date(2023, 1, 1),
        url="https://www.texasmushrooms.org/date-en/2023-01-01.htm",
    )

    photos = parse_photos(soup, day)

    assert len(photos) == 1
    p = photos[0]
    assert p.date == date(2023, 1, 1)
    assert (
        str(p.photo_url)
        == "https://www.texasmushrooms.org/pictures/archives/test_img.jpg"
    )
    assert "Found in the woods" in p.full_caption
    assert len(p.species) == 1
    assert p.species[0].latin_name == "Agaricus bisporus"
    assert (
        str(p.species[0].page_url)
        == "https://www.texasmushrooms.org/date-en/fungi_en/agaricus_bisporus.htm"
    )


def test_parse_index_handles_empty_text_links() -> None:
    # HTML snippet with empty text link but valid href date
    html = """
    <html>
    <body>
        <a href="date-en/2024-11-07.htm"></a>
        <a href="date-en/2024-11-08.htm"> </a>
    </body>
    </html>
    """

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.content = html.encode("utf-8")
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response

    days = parse_index(mock_session)

    assert len(days) == 2
    # Sorted descending
    assert days[0].date == date(2024, 11, 8)
    assert days[1].date == date(2024, 11, 7)
