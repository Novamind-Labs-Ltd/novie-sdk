from novie_prompts import client, config


def setup_function():
    config.reset()
    client.reset_client()


def test_no_connection_returns_none():
    assert client.get_client() is None


def test_test_override_takes_precedence():
    sentinel = object()
    client.set_client_for_test(sentinel)
    assert client.get_client() is sentinel


def test_override_none_is_respected():
    client.set_client_for_test(None)
    # even with a connection configured, an explicit None override wins
    config.configure(host="http://lf", public_key="pk", secret_key="sk")
    assert client.get_client() is None


def test_configure_invalidates_a_poisoned_cache(monkeypatch):
    # Fetch-before-configure: get_client caches None while unconfigured...
    assert client.get_client() is None
    # ...then configure() must invalidate that cache so a real client is built.
    built = object()
    monkeypatch.setattr(client, "_build_client", lambda conn: built)
    config.configure(host="http://lf", public_key="pk", secret_key="sk")
    assert client.get_client() is built

    # Re-configure also takes effect (not pinned to the first connection).
    built2 = object()
    monkeypatch.setattr(client, "_build_client", lambda conn: built2)
    config.configure(host="http://lf2", public_key="pk2", secret_key="sk2")
    assert client.get_client() is built2


def test_construction_failure_yields_none(monkeypatch):
    # Point at a connection but force the langfuse import/construction to blow up.
    config.configure(host="http://lf", public_key="pk", secret_key="sk")

    def _boom(*a, **k):
        raise RuntimeError("no langfuse")

    monkeypatch.setattr(client, "_build_client", _boom)
    assert client.get_client() is None  # construction failure = disabled mode = instant fallback


def test_patch_result_is_correct_regardless_of_installed_langfuse_version():
    # The CI matrix installs both langfuse 2.x (has the bug — verified against
    # a live deployment, 2.60.10) and 4.x (already fixed upstream, confirmed
    # by reading its _url_encode source: same "safe=''" comment as our fix).
    # This assertion must hold either way — patched-because-buggy or
    # left-alone-because-already-correct both end in the same place.
    from langfuse import Langfuse

    real = Langfuse(host="http://lf", public_key="pk", secret_key="sk")
    client._patch_slash_encoding_bug(real)
    assert real._url_encode("platform/session-title-system") == "platform%2Fsession-title-system"
    # non-slash names are unaffected
    assert real._url_encode("planner") == "planner"


def test_patch_is_a_noop_when_url_encode_is_absent():
    class NoUrlEncode:
        pass

    obj = NoUrlEncode()
    client._patch_slash_encoding_bug(obj)  # must not raise
    assert not hasattr(obj, "_url_encode")


def test_patch_leaves_an_already_correct_encoder_untouched():
    # Simulates the fixed-upstream case directly (independent of which
    # langfuse version happens to be installed for this test run).
    class AlreadyFixed:
        def _url_encode(self, url):
            return "SENTINEL-" + url

    obj = AlreadyFixed()
    client._patch_slash_encoding_bug(obj)
    assert obj._url_encode("/") == "SENTINEL-/"  # unpatched — our probe saw it wasn't buggy


def test_patch_leaves_an_unprobeable_encoder_untouched():
    # A method with an incompatible signature (e.g. a future major version's
    # required kwarg) must not crash the probe — patch_slash_encoding_bug
    # catches the TypeError and leaves it alone rather than guessing.
    class Incompatible:
        def _url_encode(self, url, *, required_kwarg):
            return url

    obj = Incompatible()
    client._patch_slash_encoding_bug(obj)  # must not raise
    assert obj._url_encode("/", required_kwarg=1) == "/"  # untouched, original still callable


def test_build_client_returns_a_patched_client(monkeypatch):
    # _build_client is the one real call site — verify the patch is actually wired in.
    config.configure(host="http://lf", public_key="pk", secret_key="sk")
    built = client.get_client()
    assert built is not None
    assert built._url_encode("a/b") == "a%2Fb"
