"""Atlas interativo com instruções sobre integrações do agente.

Este módulo oferece uma pequena base de conhecimento em linha de comando
que organiza instruções úteis sobre WhatsApp, Bling e utilitários
auxiliares disponíveis neste repositório.  Use-o para "abrir o atlas" e
consultar rapidamente os passos necessários.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from textwrap import fill
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class AtlasEntry:
    """Representa um tópico documentado no atlas."""

    key: str
    title: str
    description: str
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    steps: Tuple[str, ...] = field(default_factory=tuple)

    def matches(self, term: str) -> bool:
        """Retorna ``True`` se o termo ocorrer no título ou descrição."""

        term_lower = term.casefold()
        haystacks = [self.title, self.description, " ".join(self.aliases), " ".join(self.steps)]
        return any(term_lower in text.casefold() for text in haystacks if text)


_ATLAS: dict[str, AtlasEntry] = {
    "whatsapp": AtlasEntry(
        key="whatsapp",
        title="Fluxo de coleta via WhatsApp",
        description=(
            "Configure as credenciais no arquivo .env usando WHATSAPP_TOKEN e WHATSAPP_PHONE_ID. "
            "O agente deve consultar a API oficial do WhatsApp Business para ler as mensagens dos últimos 7 dias. "
            "Padronize os dados recebidos (densidade, cor, prazo) antes de gerar o orçamento."
        ),
        aliases=("zap", "wpp", "whats"),
        steps=(
            "Adicionar o token e o phone_id ao arquivo .env e validar com 'python -m http.server' se o webhook responde.",
            "Configurar o job de sincronização para buscar mensagens recentes usando o endpoint /messages.",
            "Sanitizar os dados (densidade, cor, observações) e encaminhar para o pipeline de orçamento.",
        ),
    ),
    "bling": AtlasEntry(
        key="bling",
        title="Integração com o Bling",
        description=(
            "Utilize o BLING_API_KEY para autenticar chamadas. "
            "Envie os orçamentos formatados para os endpoints de pedidos ou contatos, conforme sua operação. "
            "Verifique respostas HTTP e trate eventuais erros de validação para manter a sincronização em dia."
        ),
        aliases=("erp", "bling-erp"),
        steps=(
            "Gerar ou recuperar a chave da API no painel do Bling e armazenar em variáveis de ambiente.",
            "Montar o payload conforme a documentação /pedido/json, convertendo valores monetários para string.",
            "Registrar logs das respostas HTTP para reprocessar falhas e garantir consistência.",
        ),
    ),
    "seguranca": AtlasEntry(
        key="seguranca",
        title="Banco seguro de senhas",
        description=(
            "O utilitário password_store.py cria um banco SQLite com hashes PBKDF2-HMAC salteados. "
            "Execute 'python password_store.py add <usuario> <senha>' para cadastrar credenciais com alto custo computacional contra ataques de força bruta."
        ),
        aliases=("senhas", "password-store"),
        steps=(
            "Criar o banco com 'python password_store.py init' antes do primeiro uso.",
            "Cadastrar usuários com o comando add, usando senhas fortes e únicas.",
            "Usar o comando verify durante automações que precisam conferir credenciais sem expor texto puro.",
        ),
    ),
    "lazer": AtlasEntry(
        key="lazer",
        title="Mini-jogo Piscina da Moeda",
        description=(
            "Para uma pausa rápida, rode 'python piscina_da_moeda.py' e tente encontrar a moeda escondida em uma piscina. "
            "O jogo oferece dicas baseadas na distância euclidiana, mantendo o raciocínio afiado enquanto espera novos pedidos."
        ),
        aliases=("jogo", "moeda"),
        steps=(
            "Executar o script e escolher o tamanho da piscina.",
            "Digitar as coordenadas a cada tentativa e interpretar a dica de distância.",
            "Compartilhar a pontuação com a equipe para um intervalo rápido e saudável.",
        ),
    ),
}


def list_entries(entries: Iterable[AtlasEntry]) -> str:
    """Formata a lista de seções disponíveis."""

    lines = ["Seções disponíveis no atlas:"]
    for entry in entries:
        alias_suffix = f" (também: {', '.join(entry.aliases)})" if entry.aliases else ""
        lines.append(f"- {entry.key}: {entry.title}{alias_suffix}")
    return "\n".join(lines)


def render_entry(entry: AtlasEntry) -> str:
    """Retorna a descrição formatada de uma entrada."""

    body = fill(entry.description, width=90)
    underline = "-" * len(entry.title)
    lines = [entry.title, underline, body]
    if entry.aliases:
        lines.append(f"Também atende por: {', '.join(entry.aliases)}")
    if entry.steps:
        lines.append("")
        lines.append("Passos recomendados:")
        for idx, step in enumerate(entry.steps, 1):
            lines.append(f"{idx}. {step}")
    return "\n".join(lines)


def search_entries(term: str) -> list[AtlasEntry]:
    """Retorna entradas que contenham o termo informado."""

    return [entry for entry in _ATLAS.values() if entry.matches(term)]


def get_entry(key: str) -> AtlasEntry:
    """Recupera uma entrada pelo identificador normalizado."""

    try:
        return _ATLAS[key.casefold()]
    except KeyError as exc:  # pragma: no cover - mensagem amigável no CLI
        raise KeyError(f"Seção '{key}' não encontrada. Use --list para ver opções.") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Abre o atlas interativo com instruções rápidas sobre o agente.",
    )
    parser.add_argument(
        "--section",
        "-s",
        help="Exibe apenas a seção informada (use o identificador listado em --list).",
    )
    parser.add_argument(
        "--search",
        "-q",
        help="Pesquisa por um termo em todos os tópicos do atlas.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista as seções disponíveis antes de qualquer outra ação.",
    )
    return parser.parse_args(argv)


def build_output(args: argparse.Namespace) -> str:
    """Gera o texto apresentado ao usuário conforme os argumentos."""

    blocks: list[str] = []

    def append_block(block: str) -> None:
        if not block:
            return
        if blocks:
            blocks.append("")
        blocks.append(block)

    if args.list or (not args.section and not args.search):
        append_block(list_entries(_ATLAS.values()))

    if args.section:
        try:
            entry = get_entry(args.section)
        except KeyError as error:
            append_block(str(error))
        else:
            append_block(render_entry(entry))

    if args.search:
        matches = search_entries(args.search)
        if matches:
            block_lines = [f"Resultados para '{args.search}':"]
            for entry in matches:
                block_lines.append("")
                block_lines.append(render_entry(entry))
            append_block("\n".join(block_lines))
        else:
            append_block("Nenhuma seção corresponde ao termo buscado.")

    return "\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(build_output(args))


if __name__ == "__main__":
    main()
