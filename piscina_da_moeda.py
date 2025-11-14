"""Jogo de terminal para encontrar uma moeda escondida em uma piscina."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Posicao:
    """Representa uma posição dentro da piscina."""

    linha: int
    coluna: int

    def distancia(self, outra: "Posicao") -> float:
        """Calcula a distância euclidiana até outra posição."""

        return math.hypot(self.linha - outra.linha, self.coluna - outra.coluna)


class Piscina:
    """Piscina representada como uma grade bidimensional."""

    def __init__(self, linhas: int, colunas: int) -> None:
        if linhas <= 0 or colunas <= 0:
            raise ValueError("A piscina precisa ter dimensões positivas.")

        self.linhas = linhas
        self.colunas = colunas
        self._moeda = self._gerar_posicao_aleatoria()
        self._tentativas_restantes = max(5, (linhas * colunas) // 3)
        self._historico: List[Tuple[Posicao, float]] = []

    def _gerar_posicao_aleatoria(self) -> Posicao:
        return Posicao(random.randrange(self.linhas), random.randrange(self.colunas))

    @property
    def tentativas_restantes(self) -> int:
        return self._tentativas_restantes

    @property
    def historico(self) -> List[Tuple[Posicao, float]]:
        return list(self._historico)

    def registrar_busca(self, tentativa: Posicao) -> Tuple[bool, float]:
        """Registra uma tentativa de achar a moeda.

        Retorna uma tupla (encontrou, distancia) onde:
        - encontrou: bool indicando se a moeda foi encontrada.
        - distancia: distância até a moeda; zero se encontrada.
        """

        if self._tentativas_restantes <= 0:
            raise RuntimeError("Não há mais tentativas disponíveis.")

        self._tentativas_restantes -= 1
        distancia = tentativa.distancia(self._moeda)
        self._historico.append((tentativa, distancia))

        return distancia == 0, distancia

    def revelar_moeda(self) -> Posicao:
        return self._moeda


HINTS = (
    (2.5, "A água está bem fria por aqui... talvez esteja longe."),
    (1.5, "Você sente uma leve correnteza: a moeda deve estar mais perto."),
    (0.5, "Quase lá! Você vê um brilho prateado."),
)


def obter_hint(distancia: float) -> str:
    for limite, mensagem in HINTS:
        if distancia >= limite:
            return mensagem
    return "Está praticamente na sua mão!"


def ler_inteiro(mensagem: str, minimo: int, maximo: int) -> int:
    while True:
        try:
            valor = int(input(mensagem))
            if minimo <= valor <= maximo:
                return valor
            print(f"Digite um valor entre {minimo} e {maximo}.")
        except ValueError:
            print("Entrada inválida. Digite um número inteiro.")


def jogar() -> None:
    print("==== Jogo: Moeda na Piscina ====")
    print("A moeda caiu na piscina! Tente encontrá-la antes que o ar acabe.")

    linhas = ler_inteiro("Defina o tamanho da piscina (linhas 3-9): ", 3, 9)
    colunas = ler_inteiro("Defina o tamanho da piscina (colunas 3-9): ", 3, 9)

    piscina = Piscina(linhas, colunas)

    while piscina.tentativas_restantes > 0:
        print(f"\nTentativas restantes: {piscina.tentativas_restantes}")
        linha = ler_inteiro(f"Escolha a linha (0-{linhas - 1}): ", 0, linhas - 1)
        coluna = ler_inteiro(f"Escolha a coluna (0-{colunas - 1}): ", 0, colunas - 1)

        encontrou, distancia = piscina.registrar_busca(Posicao(linha, coluna))

        if encontrou:
            print("Você encontrou a moeda! Brilhos dourados emergem da água!")
            break

        print(f"Nada aqui. Distância até a moeda: {distancia:.2f}")
        print(obter_hint(distancia))

    else:
        posicao = piscina.revelar_moeda()
        print("\nO tempo acabou! Você precisa voltar à superfície.")
        print(f"A moeda estava na posição (linha {posicao.linha}, coluna {posicao.coluna}).")

    print("Obrigado por jogar!")


if __name__ == "__main__":
    random.seed()
    jogar()
