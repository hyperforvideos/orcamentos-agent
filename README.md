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

## Mini-jogo: Moeda na Piscina

Para relaxar enquanto espera os orçamentos chegarem, experimente o mini-jogo
`piscina_da_moeda.py`. O objetivo é encontrar a moeda perdida em uma piscina
antes que o fôlego acabe.

```
python piscina_da_moeda.py
```

Você define o tamanho da piscina e recebe pistas baseadas na distância até a
moeda a cada mergulho. Boa sorte!

Este script (a ser adicionado no futuro) irá:

- Buscar orçamentos recebidos via WhatsApp nos últimos 7 dias
- Formatar as medidas em milímetros e padronizar a formatação dos dados (densidade, cor, observações, prazo)
- Armazenar temporariamente os dados em uma planilha
- Integrar automaticamente os dados no Bling.

## Banco seguro de senhas

Para armazenar senhas de forma segura, utilize o utilitário `password_store.py`,
que cria um banco SQLite com hashes PBKDF2-HMAC salteados. Exemplos de uso:

```bash
# Cria ou atualiza um usuário
python password_store.py add alice senha_super_segura

# Verifica uma senha
python password_store.py verify alice senha_super_segura

# Lista usuários cadastrados (sem expor hashes)
python password_store.py list
```

O arquivo `credentials.db` gerado pode ser armazenado com permissões restritas
e, idealmente, em um volume criptografado. Faça backups seguros e utilize TLS
ao transmitir credenciais pela rede.

## Configuração do Visual Studio Code

### Autorização de Microfone para Conversação por Voz

Este repositório inclui configurações do VS Code (`.vscode/settings.json`) que habilitam o uso do microfone para funcionalidades de voz, como o GitHub Copilot Voice e outros assistentes por voz.

#### Funcionalidades Habilitadas:

- **Entrada de áudio**: Configurações de acessibilidade para captura de voz
- **GitHub Copilot Voice**: Suporte para comandos de voz com Copilot
- **Timeout de fala**: Configurado para 1200ms para melhor captura de comandos
- **Ativação por palavra-chave**: Configurada para chat inline

#### Como Usar:

1. Abra o repositório no Visual Studio Code
2. As configurações de microfone serão aplicadas automaticamente
3. Certifique-se de que o VS Code tem permissão para acessar seu microfone nas configurações do sistema operacional:
   - **Windows**: Configurações → Privacidade → Microfone
   - **macOS**: Preferências do Sistema → Segurança e Privacidade → Microfone
   - **Linux**: Verifique as permissões de áudio do seu ambiente

4. Para usar o GitHub Copilot Voice:
   - Instale a extensão "GitHub Copilot Voice" do marketplace
   - Use atalhos de teclado ou comandos de voz para interagir com o Copilot

#### Nota de Segurança:

As configurações de workspace habilitam confiança no workspace para permitir funcionalidades avançadas. Certifique-se de revisar o código antes de habilitar essas funcionalidades em projetos de fontes não confiáveis.
