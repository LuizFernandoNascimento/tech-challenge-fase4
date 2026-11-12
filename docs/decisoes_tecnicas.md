# Registro de decisões técnicas

Cada decisão traz **o que** foi escolhido, **por quê**, **qual aula** sustenta a
escolha e, quando houver, **o número medido** que a justifica. Este documento
alimenta o README e o roteiro do vídeo — o enunciado pede *"justificativa das
principais decisões técnicas"* (pág. 9) e *"os principais desafios e decisões
técnicas da equipe"* (pág. 10).

---

## D1 — Gemini como modelo de linguagem

**Decisão:** o LLM do pipeline é o Gemini, do Google.

**Por quê:** o enunciado delega a escolha à equipe — *"A escolha das
tecnologias, modelos, estratégias de embeddings, bancos vetoriais e técnicas de
recuperação ficará sob responsabilidade da equipe"* (pág. 6). As aulas usam
OpenAI nos exemplos de código, mas a 1IAST Aula 2 cita o stack do Google Cloud
(Vertex AI) entre as opções de RAG gerenciado. O Gemini tem camada gratuita,
o que permite ao avaliador reproduzir a solução sem custo.

---

## D2 — Versão do modelo: o que testamos e o que descobrimos

**Decisão:** `gemini-3.5-flash`.

**O que aconteceu:**

| Modelo | Resultado do teste |
|---|---|
| `gemini-2.5-flash` | **404** — aparece na listagem, mas *"no longer available to new users"* |
| `gemini-3.5-flash` | ✅ acertou o teste de abstenção em 2,1 s |
| `gemini-3.8-flash` | ✅ acertou, mas em 6,6 s — 3× mais lento sem ganho observado |
| `gemini-3.5-flash-lite` | ✅ acertou em 1,2 s |

O teste de abstenção apresentava duas avaliações que não falavam de cashback e
perguntava sobre cashback. Os três modelos disponíveis recusaram corretamente.

**Limitação de reprodutibilidade descoberta:** pedimos `gemini-3.5-flash`, mas
os erros de cota citavam `gemini-3.6-flash`. A API redireciona versões do lado
do servidor. **Fixar o nome do modelo no código não congela o modelo** — o
comportamento pode mudar sem nenhuma alteração nossa. Isso fica declarado como
limitação, não escondido.

---

## D3 — Os limites da camada gratuita (e por que mudaram a arquitetura)

**O que medimos**, no erro devolvido pela própria API:

| Limite | Valor | Identificador |
|---|---|---|
| Requisições por minuto | **5** | `GenerateRequestsPerMinutePerProjectPerModel-FreeTier` |
| Requisições por dia | **20** | `GenerateRequestsPerDayPerProjectPerModel-FreeTier` |

Ambos são **por modelo e por projeto**.

**O impacto, em números:** cada pergunta faz de 3 a 4 chamadas ao LLM

    transformação (HyDE) + julgamento de relevância + geração  [+ reescrita, se necessário]

o que dá **cerca de 5 a 6 perguntas por dia**. Na primeira bateria completa,
**7 de 15 perguntas** terminaram; as outras 8 pararam por cota diária esgotada,
não por erro no código.

**Três consequências que guiaram as decisões seguintes:**

1. **O RAGAS fica inviável na camada gratuita** — ele faz dezenas de chamadas
   por amostra para calcular cada métrica.
2. **O avaliador pode travar** ao reproduzir: com chave gratuita, o notebook
   pararia por volta da sexta pergunta.
3. **O desenvolvimento fica caro em cota** — cada teste de mudança consome uma
   fração relevante do dia.

**Quais modelos têm cota própria** (teste com uma requisição por modelo,
direto na API REST, sem novas tentativas):

| Modelo pedido | Resultado | Observação |
|---|---|---|
| `gemini-3.5-flash` | 429 | cota do dia esgotada; o balde é o do `gemini-3.6-flash` |
| `gemini-3.5-flash-lite` | ✅ | cota própria |
| `gemini-3.1-flash-lite` | ✅ | cota própria |
| `gemini-flash-lite-latest` | ✅ | respondeu como `3.5-flash-lite`: é apelido, **divide o balde** |
| `gemini-2.5-flash-lite` | 404 | indisponível para contas novas, como o `2.5-flash` |
| `gemini-3.8-flash` | timeout | inconclusivo |

Duas lições: (1) os aliases `-latest` não ampliam a cota, porque apontam para um
modelo que já está na lista; (2) há pelo menos **três baldes independentes** com
a mesma chave, o que torna o fallback entre modelos Gemini uma mitigação real.

**O que isso ensina sobre a arquitetura:** o número de chamadas ao LLM por
pergunta é uma variável de projeto, não um detalhe. Cada etapa que usa o LLM
precisa justificar o seu custo. Isso levou às mitigações da D4.

---

## D4 — Mitigações dentro da camada gratuita

Todas usam padrões ensinados nas aulas.

| Mitigação | O que faz | Aula |
|---|---|---|
| Modelo leve nas etapas auxiliares | HyDE e reescrita usam um modelo *lite*, com cota própria; julgamento e geração ficam no modelo principal | POSTECH 2 — redução de custos com modelo menor |
| `.with_fallbacks()` | Ao estourar a cota, a cadeia troca de modelo sozinha | POSTECH 2 — *"Se a API retornar Rate Limit, a chain alterna automaticamente para outro modelo"* |
| Cache de respostas do LLM | Rodar o notebook de novo não gasta cota | POSTECH 4 — caching |
| Medir o HyDE | Se não melhorar a recuperação, sai e economiza uma chamada por pergunta | — |
| Camada 1 de abstenção sem LLM | Perguntas fora do escopo são barradas **antes** de qualquer chamada | 1IAST 2 |

O último item já se pagou: as perguntas fora do escopo da bateria foram
barradas em 0,3 s **sem consumir nenhuma requisição**.

---

## D5 — Groq como provedor de fallback e juiz da avaliação

**Decisão:** o Groq entra como segundo provedor, em dois papéis: fallback quando
as cotas do Gemini se esgotam, e juiz das métricas do RAGAS.

**Por quê, e a ressalva:** o Groq **não aparece nas aulas**. A escolha se
sustenta em três pontos:

1. **O padrão é ensinado literalmente.** POSTECH Aula 2: *"Continuidade de
   Serviço: se a API da OpenAI retornar um erro 500 ou Rate Limit, a chain
   alterna automaticamente para um modelo no Azure ou Anthropic."* Fallback
   entre provedores por limite de taxa é exatamente este caso.
2. **A escolha do provedor é delegada** pelo enunciado (pág. 6).
3. **A motivação é um limite medido** (D3), não preferência.

**Limites medidos** — lidos dos cabeçalhos `x-ratelimit-*` da própria resposta:

| | Gemini (camada gratuita) | Groq (camada gratuita) |
|---|---|---|
| Requisições/dia | 20 por modelo | **1.000 por modelo** |
| Limite por minuto | 5 requisições | 8.000 tokens |
| Teste de abstenção | ✅ | ✅ nos três modelos testados |
| Latência no teste | 2,1 s | 0,2 a 0,7 s |
| Tokens/dia | — | **200.000 por modelo** (descoberto depois, ver abaixo) |

**Um limite que não vimos de início:** os cabeçalhos mostravam requisições/dia e
tokens/minuto, mas o Groq também limita **tokens por dia** (TPD): 200.000 no
`gpt-oss-120b`. Na avaliação RAGAS — que faz várias chamadas longas por resposta
(extrair afirmações, verificar cada uma, gerar perguntas, julgar cada contexto)
— estouramos esse teto antes de 1.000 requisições. O erro da API foi explícito:
*"Rate limit reached ... on tokens per day (TPD): Limit 200000"*. Consequência:
o RAGAS completo precisa ser executado num dia "limpo" de cota, e o juiz não deve
ser compartilhado com outros testes no mesmo dia. A camada gratuita tem **três**
limites a respeitar, não dois: requisições/dia, tokens/minuto e tokens/dia.

**Modelos disponíveis na conta:** `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`qwen/qwen3.8-27b`, `allam-2-7b`. Não há modelos Llama, ao contrário do que
supúnhamos. O `gpt-oss` é o modelo de pesos abertos da OpenAI — o mesmo
fornecedor dos exemplos de código das aulas.

**Por que o Groq também é o juiz do RAGAS:** além da cota, avaliar o Gemini com
um modelo de **outra família** reduz o risco de o juiz preferir respostas
parecidas com as que ele próprio geraria. É o *LLM-as-a-Judge* da 1IAST Aula 2:
*"uso de modelos de linguagem poderosos para avaliar as respostas de modelos
menores"* — um modelo de 120B julgando um modelo *flash*.

**Por que o Gemini continua como principal**, mesmo com cota 50× menor: foi a
decisão da equipe (D1), e a arquitetura não depende dessa escolha — o
provedor é uma configuração. Se a cota gratuita do Gemini for um problema na
avaliação, inverter a ordem é trocar duas linhas em `config.py`.

**Cadeia final:**

    geração e julgamento:  gemini-3.5-flash -> gemini-3.1-flash-lite -> groq gpt-oss-120b -> groq gpt-oss-20b
    HyDE e reescrita:      gemini-3.5-flash-lite -> groq gpt-oss-20b
    juiz do RAGAS:         groq gpt-oss-120b

---

## Decisões anteriores (Fase 1 e reescrita)

### D6 — Nenhum chunking
O comentário mediano tem **53 caracteres** e o maior, 208. A 1IAST Aula 2 diz
que chunking existe porque *"documentos extensos raramente cabem inteiros na
janela de contexto"* e alerta que *"chunks curtos demais podem fragmentar
contexto essencial"*. Cada avaliação já é a menor unidade de sentido.

### D7 — Dois pipelines de pré-processamento
Literal da Documento Aula 2: *"mantém duas versões do pipeline: uma agressiva
para BoW/TF-IDF, outra leve para modelos contextuais"*. Regra comum aos dois:
**negações nunca são removidas** — *"chegou"* e *"não chegou"* são a diferença
inteira numa reclamação.

### D8 — Deduplicação com contagem
4.228 avaliações repetidas colapsadas, com a contagem preservada.
*"Relatado por 47 clientes"* entra no contexto do LLM como evidência de
recorrência.

**Limite conhecido da limpeza:** o filtro de tamanho mínimo (12 caracteres) não
olha o conteúdo. Uma "avaliação" de 19 asteriscos (`*******************`,
repetida por 5 clientes) passou e virou o `doc_000000`. Varremos a base: é o
**único** documento sem nenhuma letra (outros 10 com até 8 letras são legítimos:
*"Muito bom .."*, *"Tudo ok, 100 %."*). Removê-lo renumeraria todos os
`doc_id` e exigiria retreinar os tópicos — por 1 documento em 33.332, mantivemos
e registramos. Numa próxima versão: exigir ao menos uma letra.

### D9 — Embeddings `paraphrase-multilingual-MiniLM-L12-v2`
Recomendado pela Documento Aula 4 (*"Sentence-BERT em português"*) e usado no
código da Aula 6. Indexou 33.332 documentos em **141 s**, contra 13 min do
modelo anterior.

### D10 — ChromaDB
Banco do hands-on da POSTECH Aula 3. Filtra por metadado **antes** da busca —
o que a 1IAST Aula 2 chama de *"filtragem pré-recuperação"*.

### D11 — Recuperação híbrida com pesos medidos
BM25 + densa, fundidas por RRF com c=60 (1IAST Aula 2). Pesos 0,7/0,3 escolhidos
por **varredura**, não por palpite:

| denso/léxico | 1,0/0,0 | 0,9/0,1 | **0,7/0,3** | 0,5/0,5 | 0,0/1,0 |
|---|---|---|---|---|---|
| precisão média | 0,800 | 0,833 | **0,867** | 0,733 | 0,700 |

### D12 — O Cross-Encoder ordena, mas não mede confiança
Planejamos usar a nota do Cross-Encoder para decidir quando recusar. **Medimos
e ela não serve**: *"Quais são os sintomas da gripe?"* pontuou acima de quatro
perguntas legítimas. O modelo foi treinado para *"este trecho responde à
pergunta?"*, e perguntas agregadas não são respondidas por nenhuma avaliação
isolada. Ele continua no pipeline para **ordenar** — levou a precisão em
"elogios ao atendimento" de 0,4 para 0,8.

### D13 — Abstenção em três camadas
| Camada | Pergunta | Mecanismo |
|---|---|---|
| 1. Escopo | é sobre avaliações? | similaridade do bi-encoder, **sem LLM** |
| 2. Suficiência | o recuperado sustenta resposta? | LLM julga cada evidência, reescreve, abstém |
| 3. Geração | a resposta cita o que foi recuperado? | campo `evidencia_suficiente` + validação de citações |

**Limiar da camada 1: 0,48** (configurável por `LIMIAR_ESCOPO`).

*Histórico — e um erro nosso que vale contar.* A primeira calibração usou 32
perguntas, das quais só as cinco da página 8 vinham do enunciado. Ela fixou
**0,53**: aceitava 16/16 legítimas e barrava 13/16 fora do escopo.

Na bateria de validação, uma pergunta da **página 4** do enunciado — *"Quais
problemas aparecem com maior frequência nas avaliações?"* — foi recusada como
fora do escopo: escore **0,510**, abaixo de 0,53. Perguntas genéricas
("problemas", "avaliações") têm similaridade média menor com avaliações
concretas ("o relógio veio quebrado") do que perguntas específicas. A calibração
não tinha visto esse tipo de pergunta.

**Recalibração:** 50 perguntas, agora incluindo **todas as 18 que o enunciado
cita** (págs. 4, 5, 8 e 9). As duas menores do enunciado ficam em 0,509 e 0,510.
Com **0,48**:

| | 0,53 (antigo) | **0,48 (atual)** |
|---|---|---|
| Perguntas do enunciado aceitas | 16/18 | **18/18** (margem 0,029) |
| Fora do escopo barradas na camada 1 | 13/16 | 8/16 |

**O custo** é a camada 1 barrar menos. As cinco perguntas fora do escopo que
passaram a atravessá-la (cotação do dólar, Copa do Mundo, previsão do tempo,
passagem aérea, imposto de renda) foram testadas uma a uma: **todas são barradas
pela camada 2** — o juiz não encontra avaliação relevante e o sistema se abstém.
Só gastam uma chamada a mais de LLM.

**Por que aceitamos a troca:** recusar uma pergunta do enunciado é um erro
visível para o avaliador e viola o requisito central; deixar passar uma pergunta
fora do escopo para a camada 2 custa uma chamada e termina na mesma abstenção.
A camada 1 é um filtro **barato**, não a única defesa.

**A lição:** calibrar com as perguntas que serão de fato feitas. A amostra
"representativa" que montamos não tinha perguntas genéricas de frequência.

### D14 — Gabarito de avaliação descartado
A primeira avaliação usava o título da avaliação como consulta e a mensagem como
alvo. Deu Recall@25 de 4,5%. Investigando: *"Caneta tinteiro crown"* (título) →
*"Satisfeito, recebi antes do prazo"* (mensagem). O título nomeia o produto; a
mensagem descreve a experiência. Media uma tarefa impossível, não a qualidade
da busca. Substituído por gabarito de metadados, que não estão no texto.

---

## Perguntas de frequência

### D15 — O prompt amarra termos de frequência à contagem

**O problema medido:** na primeira bateria, o modelo escreveu *"Agilidade na
entrega é o aspecto **mais frequentemente citado**"* tendo visto 10 avaliações
de 33 mil. Na segunda, *"reclamações **recorrentes** sobre os Correios"* com
base em **duas**.

**Decisão:** a regra 5 do prompt da rota RAG proíbe "recorrente", "frequente",
"comum", "maioria", "principal", "muitos" e "predominante", **a menos que** a
avaliação traga `relatada_por` com o número de clientes — e então o número deve
ser citado. Sugere no lugar "há relatos de", "aparecem casos de".

Isso corrige a desonestidade, mas **não responde** à pergunta de frequência.
Para isso existe a D16.

### D16 — Camada de tópicos com BERTopic

**Por quê:** o enunciado pergunta *"quais problemas aparecem com maior
frequência nas avaliações?"* (pág. 4) e *"quais reclamações relacionadas à
entrega aparecem com maior frequência?"* (pág. 5). O RAG recupera exemplos; não
conta. A Documento Aula 6 aponta a integração: *"BERTopic é usado para indexar e
organizar a base de conhecimento"* em pipelines de RAG.

**Como:** o pipeline da aula — embeddings → UMAP → HDBSCAN → c-TF-IDF — sobre
as 33.332 avaliações, **reaproveitando os embeddings do ChromaDB**. Cada
avaliação recebe um tópico, gravado como metadado no Chroma.

**Escolha da configuração** — seis variações testadas, com a entrada ordenada
por `doc_id` (ver D18):

| Método | Tam. mínimo | Tópicos | Sem tópico | c_v | Maior tópico |
|---|---|---|---|---|---|
| **eom** | **50** | **85** | **44,2%** | **0,622** | **6,4%** |
| eom | 100 | 34 | 31,3% | 0,558 | 27,0% |
| eom | 200 | 10 | 16,0% | 0,485 | 43,4% |
| leaf | 50 | 100 | 53,4% | 0,642 | 2,6% |
| leaf | 100 | 52 | 56,8% | 0,606 | 4,1% |
| leaf | 200 | 21 | 53,1% | 0,556 | 7,6% |

**Escolhido eom/50**, por três razões:

1. **Coerência** de 0,622 — acima do 0,5 que a aula indica como interpretável.
2. **Equilíbrio** — o maior tópico tem 6,4% das avaliações. No eom/100, um único
   tópico engole 27%; no eom/200, 43%.
3. **Estabilidade** — é praticamente igual com qualquer ordem de entrada (ver D18).

O leaf/50 tem coerência um pouco maior (0,642), mas deixa 9 pontos a mais de
avaliações sem tópico. Para **contar frequência**, cobertura pesa mais que dois
centésimos de coerência: avaliação sem tópico é avaliação que não entra na conta.

**Avaliações sem tópico.** O HDBSCAN deixa 44,2% como outliers. Elas voltam a
um tópico só se a similaridade com o centróide for **≥ 0,70**. A inspeção na
fronteira mostrou por quê:

| Similaridade | Avaliação | Tópico mais próximo | |
|---|---|---|---|
| 0,71 | *"1 dos produtos (o cavalo) não veio"* | produto não entregue | ✓ |
| 0,70 | *"Entregou antes do prazo"* | antes do prazo | ✓ |
| 0,68 | *"A barraca veio com falhas na costura"* | peça faltando | ✗ |

Com 0,70, as sem tópico caem de 44,2% para **17,5%**. Elas continuam na conta,
declaradas como "não atribuídas" — não somem.

**Rótulos.** Os 85 tópicos são nomeados pelo `gpt-oss-120b` (Groq) em lotes. Os
lotes não se veem, e o modelo **repetiu nomes**: "Produto não recebido" aparecia
em três tópicos diferentes. Uma segunda passada detecta repetições e pede nomes
que diferenciem os tópicos pelos exemplos. Resultado: 0 rótulos repetidos.

**Curiosidade dos dados:** um tópico se chama "Loja Lannister". A Olist
anonimizou os vendedores com nomes de casas de *Game of Thrones* — não é erro.

### D17 — Rota de frequência: o LLM escolhe, o código conta

**Decisão:** perguntas de frequência seguem uma rota própria, com uma regra:
**o modelo de linguagem nunca escreve um número.**

    1. o código recorta as avaliações pela polaridade - reclamação = nota 1 ou 2,
       elogio = nota 4 ou 5 - a partir do METADADO, não da interpretação do texto
    2. o código monta a tabela de tópicos com as contagens reais, somando as
       avaliações repetidas que a deduplicação colapsou
    3. para cada tópico, busca exemplos DENTRO dele, os mais próximos da pergunta
    4. o LLM escolhe QUAIS tópicos respondem à pergunta e escreve a síntese, sem números
    5. o código insere as contagens

**Três validações automáticas** em toda resposta: nenhum tópico fora da tabela,
nenhuma citação fora dos exemplos mostrados, nenhum número no texto do modelo.

**Roteamento por regra, não por LLM.** Classificar a intenção com o LLM custaria
uma chamada por pergunta, numa camada gratuita de 20 por dia (D3). É uma
*Router Chain* (POSTECH Aula 2) implementada como aresta condicional do grafo.

**Na linguagem das aulas**, é a *indexação hierárquica* da 1IAST Aula 2:
*"resumos indexados em um nível superior e chunks detalhados em um nível
inferior, combinando visão macro e micro"*. Tópicos são o nível macro (quanto);
avaliações, o micro (o quê).

**Bônus de custo:** a rota de frequência faz **1** chamada ao LLM. A rota RAG
faz de 3 a 4.

### D18 — Reprodutibilidade: a ordem da entrada muda os tópicos

**O que aconteceu:** a primeira construção deu 84 tópicos. Depois de gravarmos o
tópico de cada avaliação no Chroma, a reconstrução deu **85**, com outros
rótulos — com a mesma semente do UMAP.

**Causa:** o Chroma passou a devolver os documentos em outra ordem, e o UMAP é
sensível à ordem da entrada. Mesma semente, ordem diferente, resultado diferente.

**Correção:** os documentos são ordenados por `doc_id` antes do treino.
**Verificado:** duas execuções seguidas deram atribuições **100% idênticas** nas
33.332 avaliações.

**O que isso revelou sobre a calibração** — e uma correção nossa: na ordem
original, o eom/100 tinha dado **3 tópicos, um com 98%** das avaliações, e
chegamos a apresentá-lo como "colapso". Com a entrada ordenada, o mesmo eom/100
dá **34 tópicos**. O colapso não era propriedade da configuração — era
**instabilidade**: só a ordem dos documentos leva o eom/100 de 3 a 34 tópicos. O
eom/50 escolhido fica estável (84 ↔ 85, c_v 0,621 ↔ 0,622).

**A lição:** uma métrica de coerência não avisa quando a configuração é
instável. Foi preciso rodar duas vezes, com ordens diferentes, para ver.

### D19 — Rótulos dos tópicos levam em conta a polaridade

**O que aconteceu:** na resposta à pergunta da pág. 4 (*"quais problemas aparecem
com maior frequência?"*), apareceu entre os problemas um tópico chamado
**"Relógio entregue conforme esperado"** — com 153 avaliações de nota ≤ 2 dentro
dele. O nome era um elogio; a contagem, de reclamações.

**Causa:** o tópico é **misto**. O BERTopic agrupa por **assunto** (relógios),
não por opinião: no mesmo tópico estão *"relógio lindo, chegou rápido"* e
*"relógio veio sem a pulseira"*. O LLM nomeou o tópico pelos exemplos centrais,
que eram positivos. Medimos: **29 dos 85 tópicos** têm entre 20% e 80% de
avaliações negativas.

**Correção:** antes de rotular, calculamos o **% de avaliações negativas**
(nota ≤ 2) de cada tópico e o passamos ao LLM com uma regra:

| Classe | % negativas | O rótulo pode |
|---|---|---|
| NEGATIVO | ≥ 80% | nomear o problema ("Produto não recebido") |
| MISTO | 20–80% | **só o assunto**, sem julgamento ("Características e tamanho dos relógios") |
| POSITIVO | ≤ 20% | nomear o elogio ("Entrega antes do prazo") |

A polaridade e o % ficam gravados em `topicos_info.parquet`. Quando a pergunta
pede problemas, a rota de frequência **filtra as avaliações** por nota ≤ 2 antes
de contar (D17) — então um tópico misto aparece com a contagem só das
reclamações, e o nome neutro não contradiz essa contagem.

**Resultado:** 85 rótulos, 0 repetidos, 0 rótulos só com palavras-chave (um
tópico tinha saído como *"natal_presente_presente natal"*; agora o lote é
refeito tópico a tópico quando o modelo pula um id).

**Limite conhecido:** alguns rótulos de tópicos mistos perto da fronteira ainda
carregam leve julgamento (ex.: "Entrega atrasada além da data", 79,6% negativas).
A fronteira de 80% é uma escolha; a inspeção humana dos rótulos — que a
Documento Aula 6 chama de *insubstituível* — continua necessária.


## Validação final

### D20 — Bateria de 30 perguntas: 30/30 com o desfecho esperado

Cada pergunta tem um **desfecho esperado** declarado antes de rodar
(`scripts/bateria_testes.py`); o script confere sozinho (`scripts/rodar_bateria.py`).

| Grupo | Qtd. | Esperado | Resultado |
|---|---|---|---|
| Todas as perguntas citadas no enunciado (págs. 4, 5, 8, 9) | 18 | responder | **18/18** (11 pela rota de frequência, 7 pelo RAG) |
| Fora do escopo — barradas na camada 1, sem LLM | 2 | abster | **2/2** |
| Fora do escopo / adjacentes — passam a camada 1, camada 2 barra | 3 | abster | **3/3** |
| No tema, sem evidência (cashback, frete internacional) | 2 | abster | **2/2** |
| Adversariais de frequência ("produto mais vendido", "categorias que faturam mais") | 2 | abster | **2/2** |
| Prompt injection | 1 | bloquear | **1/1** |
| Filtros por metadado (desafio adicional) | 2 | responder | **2/2** |

**Validações automáticas da rota de frequência** (12 respostas): 0 temas
inventados, 0 citações fora do recuperado, 0 números escritos pelo LLM no texto.
Nenhum tópico de polaridade POSITIVA apareceu em respostas sobre problemas (D19).

**Observação honesta:** a bateria rodou com a cota diária do `gemini-3.5-flash`
já esgotada — todas as chamadas caíram no `gemini-3.1-flash-lite` pelo fallback
(D4/D5), e a trilha registra isso (`gemini-3.5-flash✗cota → gemini-3.1-flash-lite✓`).
Ou seja, o resultado foi obtido com o modelo **reserva**.

### D21 — HyDE: sem ganho mensurável, mantido por custo quase nulo

**Medição** (`scripts/medir_hyde.py`): 8 casos com gabarito (5 por metadado, 3
temáticos), Precisão@10, mesmo caminho do grafo (busca híbrida com a consulta,
re-ranking contra a pergunta original).

| Caso | Sem HyDE | Com HyDE |
|---|---|---|
| Categoria beleza_saude | 0,10 | 0,20 |
| Categoria cama_mesa_banho | 0,70 | 0,80 |
| Nota 1 ou 2 | **1,00** | 0,80 |
| Nota 4 ou 5 | 1,00 | 1,00 |
| Pedido realmente atrasou | 0,50 | 0,50 |
| Tema: elogios ao atendimento | 0,80 | 0,80 |
| Tema: problemas de entrega | 0,90 | 1,00 |
| Tema: qualidade do produto | 1,00 | 1,00 |
| **Média** | **0,750** | **0,762** |

**Leitura:** a diferença é de 1 documento em 80 — ruído. O HyDE ganha em 3
casos, perde em 1 e empata em 4. **Não sustentamos que o HyDE melhora a
recuperação nesta base.** Uma explicação plausível: as avaliações são curtas e
coloquiais (mediana 53 caracteres), e o re-ranking contra a pergunta original já
recupera a linguagem do cliente; a hipótese acrescenta pouco.

**Por que mantemos ligado:** (1) o custo é uma chamada à cadeia **auxiliar**,
que usa cota separada (`gemini-3.5-flash-lite` → Groq, 1.000 req/dia) — não
consome a cota de 20/dia do modelo principal; (2) torna visível a etapa de
**transformação da pergunta** que o Requisito 2 pede para demonstrar. Pode ser
desligado por pergunta: `perguntar(..., usar_hyde=False)`.

**Limite:** 8 casos é amostra pequena; o resultado indica "sem efeito
detectável", não "efeito nulo".

### D22 — RAGAS: o que medimos, dois erros de medição e uma falha real

**Como:** as 20 respostas da bateria (D20) que foram respondidas, avaliadas
com RAGAS 0.2 (`scripts/rodar_ragas.py`). Juiz: `gpt-oss-120b` no Groq —
**outro provedor** que não o Gemini que gerou as respostas. Três métricas, sem
resposta-referência (não inventamos gabarito): fidelidade (*faithfulness*),
relevância da resposta e precisão de contexto sem referência.

**Primeira medição:**

| Rota | Fidelidade | Relevância da resposta | Precisão de contexto |
|---|---|---|---|
| RAG (8 respostas) | 0,699 | 0,801 | 0,778 |
| Frequência (12) | 0,876 | 0,751 | 0,568 |
| **Geral (20)** | **0,805** | **0,771** | **0,652** |

Quatro respostas tiveram fidelidade baixa (0,00, 0,22, 0,38, 0,50). Fomos ver
uma por uma antes de aceitar o número.

**Erro de medição 1 — o juiz não via o que o gerador via.** Na resposta com
0,00, o texto diz *"Automotivo: leve e confortável [doc_022677]"*, e a avaliação
diz *"leve e super confortável"*, com `categoria="automotivo"` nos metadados.
O gerador recebe cada avaliação **com** seus metadados (nota, categoria, nº de
clientes, atraso — `formatar_contexto`); o juiz recebeu **só o texto**. Sem ver a
categoria, ele marcou como "sem suporte" toda afirmação do tipo "Categoria X:
...". O mesmo com *"compartilhado por 2 clientes"*, que vem do metadado
`relatada_por`.

**Erro de medição 2 — citações fora do campo estruturado.** Numa resposta de
frequência, o texto citava **51** avaliações, mas o campo estruturado
`ids_evidencias` listava 20 — e só essas 20 iam para o juiz. Verificamos as 37
citações "extras" da bateria: **todas existem e pertencem aos tópicos que o
modelo escolheu** (eram exemplos que ele recebeu), mas o juiz não as viu.

**Correção da medição** (`scripts/rodar_fidelidade_contexto.py`): o juiz
recebe exatamente o contexto do gerador — texto + metadados — incluindo as
avaliações citadas no texto.

| Nas mesmas 14 respostas | Fidelidade |
|---|---|
| Juiz com só o texto | 0,877 |
| **Juiz com o contexto do gerador** | **0,954** |

As respostas da rota RAG com 0,22 e 0,50 passaram a **1,00**.

**A falha real que a medição revelou:** a validação de citações (guardrail)
conferia **só o campo estruturado**. Um id inventado escrito no texto passaria.
Não aconteceu na bateria (as 37 citações extras eram válidas), mas o buraco
existia. **Corrigido:** `validar_citacoes` agora confere também todo `doc_…`
escrito no texto, nas duas rotas, e uma citação inválida é removida do texto
(*"citação removida"*) e reportada em `ids_invalidos`.

**Status — medição incompleta, e por quê:** a fidelidade com o contexto do
gerador cobre **14 de 20** respostas. As 6 restantes esbarraram de novo no teto
de **200 mil tokens/dia** do Groq (D5): as respostas de frequência levam até 60
avaliações de contexto, e cada verificação é longa. O teto é uma janela móvel
de 24 h; as 6 serão completadas quando a cota liberar
(`python -m scripts.rodar_fidelidade_contexto` — as 14 já feitas vêm do cache).

**Leitura das outras métricas:** a precisão de contexto é menor na rota de
frequência (0,568) por construção: lá o contexto são **exemplos de cada tema**
escolhido, e o juiz pergunta se cada exemplo, isoladamente, é útil para a
pergunta — um exemplo de "entrega antecipada" é pouco útil para "quais
problemas...", mesmo sendo legítimo dentro da tabela de temas.

**A lição:** um número baixo de RAGAS pode ser defeito do sistema **ou** da
medição. Aqui foram os dois — e só olhando caso a caso foi possível separar.
