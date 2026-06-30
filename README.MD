# Otimização de Pontos de Acesso WLAN Usando NSGA-II

Projeto de otimização multiobjetivo para o posicionamento de pontos de acesso (PAs) WLAN em um centro de convenções, utilizando o algoritmo evolutivo **NSGA-II** (Non-dominated Sorting Genetic Algorithm II).

## Autores

- Arthur Lima Fernandes — Universidade Federal de Minas Gerais — arthurlkf@ufmg.br
- Cleyton Luan Nobre Assis — Universidade Federal de Minas Gerais — cleyton-nobre@ufmg.br
- Mateus Vilela Oliveira — Universidade Federal de Minas Gerais — mateus.vilela025@gmail.com

## Sobre o projeto

Este trabalho aborda a otimização da instalação de pontos de acesso WLAN em um centro de convenções com área de **400m x 400m**, discretizado em pontos de instalação possíveis a cada 5m (totalizando 6400 posições candidatas).

O problema é tratado como um **problema de atribuição multiobjetivo**, com dois objetivos conflitantes:

1. **Minimizar o número total de pontos de acesso instalados**
2. **Minimizar a distância total entre os pontos de acesso e os clientes atendidos**

Para resolvê-lo, foi escolhido o algoritmo evolutivo **NSGA-II**, reconhecido por sua eficiência na aproximação da Fronteira de Pareto com baixo custo computacional e boa manutenção da diversidade das soluções.

## Modelagem do problema

O problema é formulado com variáveis de decisão binárias:

- `x_ij`: igual a 1 se o cliente *j* é atendido pelo ponto de acesso *i*
- `y_i`: igual a 1 se o ponto de acesso *i* está ativo

### Funções objetivo

- **f1(x, y)** — soma das distâncias entre cada ponto de acesso ativo e os clientes que ele atende
- **f2(y)** — número total de pontos de acesso ativos

### Principais restrições

- Atendimento mínimo de clientes (taxa de atendimento η)
- Capacidade máxima de cada ponto de acesso (q)
- Raio de cobertura do sinal de cada ponto de acesso (r_i)
- Cada cliente atendido por no máximo um ponto de acesso
- Número máximo de pontos de acesso permitidos (m_max)

### Parâmetros do problema

| Parâmetro | Valor |
|---|---|
| Número de clientes | 495 |
| Posições possíveis de PAs | 6400 |
| Taxa de atendimento mínima | 98% |
| Capacidade por ponto de acesso | 54 Mbps |
| Distância máxima cliente–PA | 85 m |
| Número máximo de PAs permitidos | 30 |

**Premissas adotadas:** pontos de demanda e consumo estáticos, obstáculos internos desprezíveis, e ausência de interferência entre pontos de acesso.

## Metodologia — NSGA-II

O NSGA-II foi proposto em 2002 por Deb et al. como uma evolução de algoritmos genéticos multiobjetivo anteriores, trazendo três melhorias principais:

- Introdução de **elitismo**
- Redução da complexidade computacional de O(MN³) para **O(MN²)**
- Remoção do parâmetro de compartilhamento (*sharing*), antes necessário para preservar diversidade

### Componentes principais

- **Índice de Fronteira (I_f):** classifica as soluções por dominância, atribuindo índice 0 às soluções não dominadas, depois às seguintes camadas
- **Distância de Aglomeração (δ_a):** mede a diversidade de uma solução em relação às suas vizinhas no espaço de objetivos, sendo usada como critério de desempate

### Representação das soluções

Cada solução é representada por um vetor binário `Y = [y1, y2, ..., yn]`, em que cada posição corresponde a um ponto candidato no mapa, varrido da esquerda para a direita e de cima para baixo, totalizando 6400 elementos.

### Operadores genéticos

- **Seleção:** torneio binário (dominância + distância de aglomeração)
- **Crossover:** ponto de corte único, aleatório
- **Mutação:** bit-flip

### Etapas do algoritmo

1. Inicialização da população
2. Avaliação das funções objetivo
3. Classificação por dominância (fronteiras de Pareto)
4. Cálculo da distância de aglomeração
5. Seleção, crossover e mutação
6. Atualização e truncamento da população (mantendo tamanho N)

O ciclo é repetido por G gerações, resultando em um conjunto de soluções não dominadas que aproxima a Fronteira de Pareto.

## Avaliação dos resultados

A qualidade das soluções é avaliada pela métrica de **hipervolume** (s-metric), que mede simultaneamente convergência e diversidade do conjunto de soluções não dominadas, em relação a um vetor de referência anti-utópico (nadir).

## Referências

1. A. Gaspar-Cunha, R. Takahashi e C. H. Antunes, *Manual de Computação Evolutiva e Metaheurística*. Editora UFMG, 2013.
2. K. Deb, A. Pratap, S. Agarwal e T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", *IEEE Transactions on Evolutionary Computation*, vol. 6, n.º 2, pp. 182–197, 2002.
3. A. Auger, J. Bader, D. Brockhoff e E. Zitzler, "Hypervolume-based multiobjective optimization: Theoretical foundations and practical implications", *Theoretical Computer Science*, vol. 425, pp. 75–103, 2012.