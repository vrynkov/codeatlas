"""What the public server agrees to clone. Nothing here runs git or touches GitHub."""
import pytest

from app.indexing import indexer
from app.indexing.indexer import CloneError, parse_github_url


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/psf/requests", ("psf", "requests")),
    ("https://github.com/psf/requests.git", ("psf", "requests")),
    ("https://github.com/psf/requests/", ("psf", "requests")),
    ("http://github.com/psf/requests", ("psf", "requests")),
    ("github.com/vrynkov/codeatlas", ("vrynkov", "codeatlas")),
    ("https://github.com/a-b/my_repo.v2", ("a-b", "my_repo.v2")),
])
def test_valid_github_urls(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",                          # read the server's own disk
    "https://gitlab.com/a/b",                      # other hosts
    "https://github.com.evil.com/a/b",             # look-alike host
    "https://evil.com/github.com/a/b",
    "ssh://git@github.com/a/b",
    "git@github.com:a/b.git",
    "--upload-pack=touch /tmp/pwned",              # git option injection
    "https://github.com/-oops/repo",
    "https://github.com/a/..",                     # path tricks
    "https://github.com/a/b/../../c",
    "https://github.com/a/b?x=1",
    "https://github.com/a/b c",
    "https://user:pass@github.com/a/b",
    "http://169.254.169.254/latest/meta-data",     # cloud metadata server
    "",
])
def test_everything_else_is_refused(url):
    with pytest.raises(CloneError):
        parse_github_url(url)


@pytest.fixture
def no_git(tmp_path, monkeypatch):
    """Records git commands instead of running them; GitHub's size API is faked as 'unknown'."""
    calls = []
    monkeypatch.setattr(indexer, "REPOS_DIR", tmp_path / "repos")
    monkeypatch.setattr(indexer, "_git", lambda *args: calls.append(args))
    monkeypatch.setattr(indexer, "_github_size_mb", lambda owner, repo: None)
    return calls


def test_clone_uses_a_rebuilt_url_after_a_double_dash(no_git):
    indexer.clone_repo("github.com/psf/requests/")
    args = no_git[0]
    assert args[:3] == ("clone", "--depth", "1")
    assert args[3] == "--" and args[4] == "https://github.com/psf/requests.git"


def test_bad_url_never_reaches_git(no_git):
    with pytest.raises(CloneError):
        indexer.clone_repo("file:///etc")
    assert no_git == []


def test_too_big_by_githubs_numbers_is_refused_before_download(no_git, monkeypatch):
    monkeypatch.setattr(indexer, "_github_size_mb", lambda owner, repo: indexer.MAX_REPO_MB + 1)
    with pytest.raises(CloneError, match="MB"):
        indexer.clone_repo("https://github.com/big/monorepo")
    assert no_git == []


def test_too_big_after_download_is_deleted(no_git, monkeypatch):
    monkeypatch.setattr(indexer, "_folder_mb", lambda path: indexer.MAX_REPO_MB + 1)
    with pytest.raises(CloneError, match="after download"):
        indexer.clone_repo("https://github.com/sneaky/repo")
    assert not (indexer.REPOS_DIR / "repo").exists()


def test_same_name_from_another_owner_cannot_replace_an_indexed_repo(no_git, monkeypatch):
    (indexer.REPOS_DIR / "requests").mkdir(parents=True)
    monkeypatch.setattr(indexer, "repo_source_url", lambda name: "https://github.com/psf/requests.git")
    with pytest.raises(CloneError, match="different repo"):
        indexer.clone_repo("https://github.com/evil/requests")
    assert no_git == []
    indexer.clone_repo("https://github.com/PSF/requests")  # the SAME repo may be refreshed
    assert no_git[0][:2] == ("-C", str(indexer.REPOS_DIR / "requests"))


def test_only_one_indexing_job_at_a_time(monkeypatch):
    monkeypatch.setitem(indexer._PROGRESS, "other", {"stage": "embedding"})
    with pytest.raises(indexer.IndexBusyError):
        indexer.start_indexing_job("https://github.com/psf/requests")


def test_repo_with_too_many_chunks_is_refused(no_git, monkeypatch):
    monkeypatch.setattr(indexer, "scan_repo", lambda path: {"file_count": 1, "total_bytes": 1})
    monkeypatch.setattr(indexer, "chunk_repo", lambda path: [object()] * (indexer.MAX_REPO_CHUNKS + 1))
    monkeypatch.setattr(indexer, "_indexed_before", lambda name: False)
    with pytest.raises(CloneError, match="code pieces"):
        indexer.start_indexing_job("https://github.com/huge/repo")
