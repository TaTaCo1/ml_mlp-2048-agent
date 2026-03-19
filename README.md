# 2048 Agent: DQN & MCTS

Este repositório contém o código para treinar, jogar e avaliar um Agente de Reforço (Reinforcement Learning) Inteligente para o jogo 2048, usando uma combinação de redes neurais **DQN (Deep Q-Network)** e o algoritmo de planejamento **MCTS (Monte Carlo Tree Search)**.

Aqui você encontra instruções rápidas de como rodar cada parte do projeto.

---

## 1. Como Treinar o Agente (`train.py`)

Você pode treinar o agente de duas formas na sua máquina ou servidor:
1. **Treinamento padrão (Apenas Reflexo via DQN):** Ideal para construir a intuição rápida do agente. O aprendizado ocorre instantaneamente a cada movimento.
2. **Treinamento via simulações MCTS:** O agente usará planejamento MCTS *durante* o treinamento (muito mais lento computacionalmente, mas incrivelmente preciso a cada passo).

### Módulo DQN (Rápido e Padrão)
Para iniciar o treinamento base por 1000 episódios:
```bash
python train.py --episodes 1000
```
Se você parou o treino no meio e quer retomar de onde parou usando os últimos pesos e métricas:
```bash
python train.py --episodes 500 --resume
```

### Módulo DQN + MCTS (Treinamento Planejado Analítico)
Para treinar acionando o MCTS durante as partidas:
```bash
python train.py --episodes 500 --use-mcts --mcts-sims 20 --mcts-depth 30
```
- `--mcts-sims`: Quantidade de simulações mentais por movimento (padrão 20).
- `--mcts-depth`: Limite de profundidade das ramificações (padrão 30).

**Onde os pesos do agente ficam salvos?**
Ele salva os pesos em `dqn_latest.pth` / `dqn_best.pth` se não usar MCTS. 
Com MCTS, ele salva como `dqn_mcts_latest.pth` e `dqn_mcts_best.pth`. As estatísticas são registradas nos arquivos de métricas (JSON correspondentes).

---

## 2. Como Testar e Avaliar os Agentes (`evaluate.py`)

Depois que o modelo está treinado, é hora de avaliar e comparar quem joga melhor o 2048. O arquivo `evaluate.py` testa o agente e gera arquivos `.json` com a distribuição das melhores peças de cada rodada, bem como o ranking dos melhores jogos.

### Comparar Todo Mundo (DQN vs MCTS vs Random)
Deixa o avaliador testar os 3 modos de jogo seguidos (cada um batendo cabeça por 100 partidas) e imprimir uma tabela de comparação final.
```bash
python evaluate.py --mode all --episodes 100
```
> **Aviso:** Testar o MCTS pode demorar consideravelmente em centenas de episódios, pois ele fará milhares de clonagens do jogo.

### Testar Apenas Um Agente Específico
Se você quer ver exclusivamente como o DQN simples está jogando:
```bash
python evaluate.py --mode dqn --episodes 50
```

Se quiser ver apenas o resultado esplêndido (porém super demorado) do MCTS usando a versão avançada do seu modelo (é necessário que exista o `dqn_mcts_latest.pth` na pasta, ou apontar via parâmetro `--model-mcts`):
```bash
python evaluate.py --mode mcts --episodes 10
```

---

## 3. Como Plotar os Gráficos (`plots.py`)

Os gráficos ajudam a ilustrar visualmente quem ganha mais vezes e como se deu a evolução matemática (Loss/Reward) durante as horas de treinamento.

Para gerar e salvar todos os gráficos na pasta, basta rodar:
```bash
python plots.py
```
Esse comando lerá seus arquivos `.json` que foram preenchidos (tanto pelo `.train` quanto pelo `evaluate.py`) e exportará imagens contendo:
- O histórico de Treinamento (Scores, Recompensas por época, e Loss da Rede).
- O gráfico de pilar incrível mostrando as Distribuições de Pontuação (ex: O MCTS chegou até no `2048` na barra gráfica vermelha).

Você pode alterar os nomes dos arquivos lidos pelo `plots.py` caso você treine agentes customizados modificando os alertas:
```bash
python plots.py --metrics_dqn seu_arquivo_de_metrica.json
```
