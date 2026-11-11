"""Conjunto de calibração do limiar de escopo (camada 1 da abstenção).

DENTRO: perguntas que avaliações de clientes conseguem responder - incluindo
as cinco do enunciado (pág. 8) e variações por tema, produto e tom.
FORA: perguntas que avaliações não respondem. Inclui casos ADJACENTES ao
e-commerce de propósito ("faturamento da Olist", "como abrir uma loja") -
são os mais difíceis, porque compartilham vocabulário com a base.
"""
# TODAS as perguntas que o enunciado cita. Nenhuma pode ser barrada pela camada 1.
ENUNCIADO = [
    # página 4
    "Quais são os principais motivos de insatisfação dos clientes?",
    "Quais problemas aparecem com maior frequência nas avaliações?",
    "O que os consumidores mais valorizam em sua experiência?",
    "Quais padrões podem ser identificados em relatos relacionados à entrega, qualidade dos produtos ou atendimento?",
    "Quais evidências presentes nos comentários sustentam determinado insight?",
    # página 5
    "Quais são os principais problemas relatados pelos clientes?",
    "Quais reclamações relacionadas à entrega aparecem com maior frequência?",
    "O que os clientes mais elogiam em suas experiências de compra?",
    "Quais problemas são relatados por clientes insatisfeitos?",
    "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
    # página 8
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    # página 9 (desafio adicional)
    "Quais são os principais problemas relatados nas avaliações com nota baixa?",
    "O que os clientes mais elogiam em determinadas categorias de produtos?",
    "Quais temas aparecem nas avaliações relacionadas a experiências de entrega insatisfatórias?",
]

DENTRO = [
    # as cinco investigações da página 8 do enunciado
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    # variações
    "Os clientes reclamam de produtos que chegam quebrados?",
    "O que dizem sobre o atendimento do vendedor?",
    "Há reclamações sobre produto diferente do anunciado?",
    "Os clientes elogiam a embalagem?",
    "O que reclamam sobre a qualidade dos produtos?",
    "Há relatos de pedidos que nunca chegaram?",
    "Os clientes recomendariam a loja?",
    "Como os clientes avaliam o custo-benefício?",
    "Existem reclamações sobre itens faltando no pedido?",
    "O que os clientes dizem sobre roupas de cama?",
    "Há queixas sobre a falta de resposta do vendedor?",
]
FORA = [
    "Qual é a cotação do dólar hoje?",
    "Quem ganhou a Copa do Mundo de 2022?",
    "Como fazer um bolo de cenoura?",
    "Qual é a capital da Austrália?",
    "Quais são os sintomas da gripe?",
    "Como declarar imposto de renda?",
    "Qual a melhor linguagem de programação?",
    "Quanto custa uma passagem para Portugal?",
    "Qual a previsão do tempo para amanhã?",
    "Quem escreveu Dom Casmurro?",
    "Como funciona a fotossíntese?",
    "Quais são as regras do xadrez?",
    # adjacentes ao e-commerce - os casos difíceis
    "Qual foi o faturamento da Olist em 2018?",
    "Como abrir uma loja na Olist?",
    "Quantos funcionários a Olist tem?",
    "Qual a taxa de juros do cartão de crédito?",
]
