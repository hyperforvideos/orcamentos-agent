from types import SimpleNamespace

from atlas import (
    AtlasEntry,
    build_output,
    get_entry,
    list_entries,
    render_entry,
    search_entries,
)


def test_list_entries_includes_all_sections():
    output = list_entries([
        AtlasEntry("demo", "Demo", "Texto", aliases=("alias-demo",)),
        AtlasEntry("teste", "Teste", "Outro"),
    ])
    assert "demo" in output
    assert "teste" in output
    assert "alias-demo" in output


def test_get_entry_returns_expected_section():
    entry = get_entry("whatsapp")
    assert entry.title.startswith("Fluxo")


def test_search_entries_finds_term():
    matches = search_entries("WhatsApp")
    assert matches
    assert any(entry.key == "whatsapp" for entry in matches)


def test_search_entries_finds_aliases():
    matches = search_entries("wpp")
    assert any(entry.key == "whatsapp" for entry in matches)


def _args(**kwargs):
    defaults = {"list": False, "section": None, "search": None}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_output_lists_sections_by_default():
    output = build_output(_args())
    assert "Seções disponíveis" in output


def test_build_output_includes_specific_section_when_requested():
    output = build_output(_args(section="whatsapp"))
    assert "Fluxo de coleta via WhatsApp" in output
    assert "Passos recomendados" in output


def test_build_output_reports_error_for_unknown_section():
    output = build_output(_args(section="inexistente"))
    assert "Seção" in output
    assert "não encontrada" in output


def test_build_output_search_lists_matches():
    output = build_output(_args(search="Bling"))
    assert "Resultados para" in output
    assert "Integração com o Bling" in output


def test_render_entry_includes_steps_and_aliases():
    entry = AtlasEntry(
        "demo",
        "Demo",
        "Descrição",
        aliases=("alias",),
        steps=("Primeiro passo", "Segundo passo"),
    )
    output = render_entry(entry)
    assert "alias" in output
    assert "Passos recomendados" in output
    assert "1. Primeiro passo" in output
