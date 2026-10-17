from hauserkennen.__main__ import main

AREA = ["--lat", "47.08432", "--lon", "15.45938", "--radius", "100"]


def test_missing_photo_is_reported_before_any_download(tmp_path, capsys):
    code = main(["suche", "--foto", str(tmp_path / "fehlt.jpg"), *AREA, "--cache", str(tmp_path / "c")])
    assert code == 2
    assert "Foto nicht gefunden" in capsys.readouterr().err


def test_missing_google_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    photo = tmp_path / "f.jpg"
    photo.write_bytes(b"x")
    code = main(["suche", "--foto", str(photo), *AREA, "--google-key", "", "--cache", str(tmp_path / "c")])
    assert code == 2
    assert "API-Key" in capsys.readouterr().err
