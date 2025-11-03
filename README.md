# orcamentos-agent
Agente de automação para coletar orçamentos do WhatsApp, formatá-los e integrar ao Bling.

## Instalação

Clone este repositório e crie um ambiente virtual (exemplo usando Python):

```
bash
# Clone o repositório
 git clone https://github.com/hyperforvideos/orcamentos-agent.git
 cd orcamentos-agent

# Crie um ambiente virtual e ative
 python -m venv .venv
 source .venv/bin/activate  # no Windows use `.venv\Scripts\activate`

# Instale dependências (adicione requirements.txt ao repositório conforme necessário)
 pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` com as credenciais necessárias:

```
BLING_API_KEY=coloque_sua_chave_aqui
WHATSAPP_TOKEN=seu_token_de_acesso
WHATSAPP_PHONE_ID=seu_phone_id
```

## Execução

Com o ambiente configurado, execute o agente para coletar orçamentos e integrá-los ao Bling:

```
python orcamentos_agent.py
```

Este script (a ser adicionado no futuro) irá:

- Buscar orçamentos recebidos via WhatsApp nos últimos 7 dias
- Formatar as medidas em milímetros e padronizar a formatação dos dados (densidade, cor, observações, prazo)
- Armazenar temporariamente os dados em uma planilha
- Integrar automaticamente os dados no Bling.
