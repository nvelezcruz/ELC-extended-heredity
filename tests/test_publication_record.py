from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "publication"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_final_pdfs_match_the_supplied_publication_files() -> None:
    assert _digest(
        PUBLICATION / "manuscript" / "An_Information_Theory_of_Extended_Heredity.pdf"
    ) == "fb72dabaef7050e378a15e2a767e49dc185575919607f741a4dd7156216fe18f"
    assert _digest(
        PUBLICATION / "supplement" / "Supplementary_Material.pdf"
    ) == "0d5764025d1165a5a480bfdc019c753520ce0e197c0100c7463a8124bf4de629"


def test_publication_sources_report_the_final_c13_results() -> None:
    manuscript = (PUBLICATION / "manuscript" / "main.tex").read_text()
    supplement = (PUBLICATION / "supplement" / "main.tex").read_text()
    for text in (manuscript, supplement):
        assert "$0.284$ bits" in text
        assert "$0.0867$ bits" in text
        assert "$0.0298$ bits" in text
        assert "$0.0110$ bits" in text
        assert "$0.0274$ bits" in text
    assert "$455$ of the $1{,}000$ ELC realizations ($45.5\\%$)" in manuscript
    assert "Predictive contribution and dependence on the ELC" in supplement
