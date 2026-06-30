import numpy as np
import random
import matplotlib.pyplot as plt
from solucaoInicial import gerar_resultados
import copy

class NSGAII:
    def __init__(self, PAs:list, users:list, pop_size:int=100, max_gen:int=100, 
                 crossover_rate:float=0.9, mutation_rate:float=0.1, 
                 tournament_size:int=2, max_pas:int=30, total_users:int=495,
                 verbose:bool=False, debug:bool=False):
        """
        Inicializa o algoritmo NSGA-II.

        Args:
            PAs (list): Lista de PAs (objeto Ponto_Acesso) inicializados.
            users (list): Lista de usuários (objeto Usuario) inicializados.
            pop_size (int): Tamanho da população.
            max_gen (int): Número máximo de gerações.
            crossover_rate (float): Taxa de crossover.
            mutation_rate (float): Taxa de mutação.
            tournament_size (int): Tamanho do torneio.
            max_pas (int): Número máximo de PAs.
            total_users (int): Número total de usuários.
            verbose (bool): Se True, exibe informações sobre a execução do algoritmo.
            debug (bool): Se True, exibe informações de debug.
        """
        self.PAs = PAs
        self.users = users
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.max_pas = max_pas
        self.total_users = total_users
        self.verbose = verbose
        self.debug = debug
        
        # Configurações específicas do problema
        self.num_PAs = len(PAs)
        self.PA_bounds = [0, 1]  # 0 para PA desativado, 1 para PA ativado
        
        # Referência para hipervolume (vetor anti-utópico)
        self.reference_point = [0, 0]
        
        # Cache para verificações de restrições
        self.constraint_violation_cache = {}  # Dicionário para armazenar resultados de restrições por geração

    def clear_constraint_cache(self):
        """
        Limpa o cache de violação de restrições para uma nova geração.
        """
        self.constraint_violation_cache.clear()

    def update_reference_point(self, objectives:list)->None:
        """
        Atualiza o ponto de referência com os valores máximos encontrados.

        Args:
            objectives (list): Lista de tuplas com resultados das funções objetivo para cada solução.
        """
        for obj in objectives:
            self.reference_point[0] = max(self.reference_point[0], obj[0])
            self.reference_point[1] = max(self.reference_point[1], obj[1])

    def calculate_hypervolume(self, pareto_front:list)->float:
        """
        Calcula o hipervolume do conjunto de soluções não dominadas
        usando o algoritmo de Zitzler e Thiele

        Args:
            pareto_front (list): Lista de tuplas com resultados das funções objetivo para cada solução.

        Returns:
            float: Hipervolume do conjunto de soluções não dominadas.
        """
        # Ordena as soluções pelo primeiro objetivo
        # Ordena as soluções pelo primeiro objetivo
        sorted_front = sorted(pareto_front, key=lambda x: x[0])

        # Calcula o hipervolume das soluções
        hv = 0.0
        previous_f1 = self.reference_point[0]

        for f1, f2 in reversed(sorted_front):
            width = previous_f1 - f1
            height = self.reference_point[1] - f2
            hv += width * height
            previous_f1 = f1

        # Calcula o hipervolume total possível
        min_f1 = min(p[0] for p in pareto_front)
        min_f2 = min(p[1] for p in pareto_front)
        hv_total = (self.reference_point[0] - min_f1) * (self.reference_point[1] - min_f2)

        # Porcentagem do hipervolume
        hv_percentage = (hv / hv_total) * 100 if hv_total != 0 else 0.0
        return hv_percentage

    def check_constraints(self, solution:np.ndarray, objectives:tuple=None)->bool:
        """
        Verifica se a solução atende todas as restrições do problema.
        - Restrição 1: garante o atendimento de clientes de acordo com a taxa de atendimento mínima.
        - Restrição 2: define que o consumo dos clientes atendidos não pode ultrapassar a capacidade do PA, se ele estiver ativo. 
        - Restrição 3: garante que um cliente j atendido por um PA i deve estar dentro do raio de cobertura ri deste PA, se o PA estiver ativo. 
        - Restrição 4: estabelece que cada cliente j deve ser atendido por, no máximo, um único PA 
        - Restrição 5: define que o número de PAs ativos deve ser menor ou igual ao número máximo de PAs aceitos. 
        
        Args:
            solution (np.ndarray): Vetor binário representando uma solução.
            objectives (tuple, opcional): Tupla com resultados das funções objetivo para a solução. 
                Usado apenas para debug, quando fornecido, os valores serão exibidos nas mensagens de debug.
                Se None, exibirá "Solução" nas mensagens de debug.

        Returns:
            bool: True se a solução é factível, False caso contrário.
        """
        # Converte solução para tupla para usar como chave no cache
        solution_tuple = tuple(solution)
        
        # Verifica se já temos o resultado em cache
        if solution_tuple in self.constraint_violation_cache:
            return self.constraint_violation_cache[solution_tuple]
        
        if objectives is None:
            objectives = "Solução"
        
        # Verifica número máximo de PAs (Restrição 5)
        if sum(solution) > self.max_pas:
            if self.debug: print(f"DEBUG: RESTRICOES - {objectives} viola Restrição 5: {sum(solution)} > {self.max_pas}")
            self.constraint_violation_cache[solution_tuple] = False
            return False

        # Verifica atendimento mínimo (Restrição 1)
        PAs_ativos, users_atendidos = gerar_resultados(solution, self.PAs, self.users)
        atendidos = sum(1 for u in users_atendidos if u.user_atendido)
        min_atendimento = int(0.98 * self.total_users)
        if atendidos < min_atendimento:
            if self.debug: print(f"DEBUG: RESTRICOES - {objectives} viola Restrição 1: {atendidos} < {min_atendimento}")
            self.constraint_violation_cache[solution_tuple] = False
            return False

        # Verifica capacidade e distância
        for pa in PAs_ativos:
            if pa.banda_disponivel < 0:  # Capacidade excedida (Restrição 2)
                if self.debug: print(f"DEBUG: RESTRICOES - {objectives} viola Restrição 2: {pa.banda_disponivel} < 0")
                self.constraint_violation_cache[solution_tuple] = False
                return False
            for user in pa.usuarios_atendidos:
                distancia = round(np.linalg.norm(np.array(user.coordenadas) - np.array(pa.coordenadas)),2)
                if distancia > pa.raio:  # Distância excedida (Restrição 3)
                    if self.debug: print(f"DEBUG: RESTRICOES - {objectives} viola Restrição 3: {distancia} > {pa.raio}")
                    self.constraint_violation_cache[solution_tuple] = False
                    return False

        # A Restrição 4 é implícita na forma como o problema é formulado, uma vez que 
        # cada objeto Usuario tem um atributo user_atendido. Este atributo booleano 
        # impede que um usuario seja atendido por mais de um PA.

        # Solução é factível
        self.constraint_violation_cache[solution_tuple] = True
        if self.debug: print(f"DEBUG: RESTRICOES - {objectives} não viola nenhuma restrição")
        return True

    def dominates(self, a:tuple, b:tuple, solution_a:np.ndarray=None, solution_b:np.ndarray=None)->bool:
        """
        Verifica se solução a domina solução b, considerando restrições.

        Args:
            a (tuple): Tupla com resultados das funções objetivo para solução a.
            b (tuple): Tupla com resultados das funções objetivo para solução b.
            solution_a (np.ndarray): Solução a (opcional).
            solution_b (np.ndarray): Solução b (opcional).

        Returns:
            bool: True se a domina b, False caso contrário.
        """
        # Se a solução a é inválida, ela não domina ninguém
        if solution_a is not None and not self.check_constraints(solution_a, a):
            return False
        # Se a solução b é inválida, ela é dominada por a
        if solution_b is not None and not self.check_constraints(solution_b, b):
            return True

        # Se ambas são válidas, usa a dominância normal
        return (all(x <= y for x, y in zip(a, b)) and 
                any(x < y for x, y in zip(a, b)))

    def format_debug_message(self, p: int, q: int, obj_p, obj_q, relacao: str) -> str:
        """
        Formata uma mensagem de debug para relações de dominância.
        
        Args:
            p (int): Índice da primeira solução
            q (int): Índice da segunda solução
            obj_p: Objetivos da solução p
            obj_q: Objetivos da solução q
            relacao (str): Tipo de relação (domina, dominada por, equivalente a)
            
        Returns:
            str: Mensagem formatada
        """
        maior_relacao = "[equivalente a]"

        # Formatação consistente para índices e valores
        idx_p = f"{p:3d}"
        idx_q = f"{q:3d}"
        obj_p = f"({obj_p[0]:.2f}, {obj_p[1]:3d})".ljust(15)
        obj_q = f"({obj_q[0]:.2f}, {obj_q[1]:3d})".ljust(15)
        relacao = f"{relacao}".ljust(len(maior_relacao))
        
        return f"DEBUG: DOMINACAO - Solução {idx_p} {obj_p}  {relacao}  Solução {idx_q} {obj_q}"

    def non_dominated_sort(self, population:list, objectives:list)->list:
        """
        Ordena a população em frentes de Pareto.

        Args:
            population (list): Lista de soluções.
            objectives (list): Lista de tuplas com resultados das funções objetivo para cada solução.

        Returns:
            list: Lista de frentes de Pareto.
        """
        # Inicializa as estruturas de dados
        fronts = []
        domination_count = [0] * len(population)
        dominated = [[] for _ in population]

        # Para cada solução p
        for p in range(len(population)):
            for q in range(p + 1, len(population)):
                domination_check = False
                # Se p domina q
                if self.dominates(objectives[p], objectives[q], 
                               solution_a=population[p], solution_b=population[q]):
                    dominated[p].append(q)
                    domination_count[q] += 1
                    domination_check = True
                    if self.debug: print(self.format_debug_message(p, q, objectives[p], objectives[q], "[domina]"))
                # Se q domina p
                elif not domination_check and self.dominates(objectives[q], objectives[p], 
                                 solution_a=population[q], solution_b=population[p]):
                    dominated[q].append(p)
                    domination_count[p] += 1
                    if self.debug: print(self.format_debug_message(p, q, objectives[p], objectives[q], "[dominada por]"))
                else:
                    if self.debug: print(self.format_debug_message(p, q, objectives[p], objectives[q], "[equivalente a]"))
            
            # Se p não é dominado por ninguém
            if domination_count[p] == 0:
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(p)
        
        # Para cada frente
        i = 0
        while True:
            if i >= len(fronts):
                break
            
            next_front = []
            for p in fronts[i]:
                for q in dominated[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            i += 1
        
        return fronts

    def crowding_distance(self, front:list, objectives:list)->list:
        """
        Calcula a distância de aglomeração para manter diversidade.

        Args:
            front (list): Lista de soluções em uma frente de Pareto.
            objectives (list): Lista de tuplas com resultados das funções objetivo para cada solução.

        Returns:
            list: Lista de distâncias de aglomeração.
        """
        solution_distances = [0.0] * len(front)
        for objective_idx in range(len(objectives[0])):
            obj_vals = [objectives[solution_idx][objective_idx] for solution_idx in front]
            sorted_indices = np.argsort(obj_vals)
            f_min = obj_vals[sorted_indices[0]]
            f_max = obj_vals[sorted_indices[-1]]
            
            # Primeiro e último têm distância infinita
            solution_distances[sorted_indices[0]] = float('inf')
            solution_distances[sorted_indices[-1]] = float('inf')
            
            # Calcula distância para os demais
            for solution_idx in range(1, len(front) - 1):
                prev_val = obj_vals[sorted_indices[solution_idx - 1]]
                next_val = obj_vals[sorted_indices[solution_idx + 1]]
                if f_max - f_min == 0:
                    d = 0
                else:
                    d = (next_val - prev_val) / (f_max - f_min)
                solution_distances[sorted_indices[solution_idx]] += d
                
        return solution_distances

    def tournament_selection(self, population:list, objectives:list)->np.ndarray:
        """
        Seleção por torneio binário baseado em dominância.

        Args:
            population (list): Lista de soluções.
            objectives (list): Lista de tuplas com resultados das funções objetivo para cada solução.

        Returns:
            np.ndarray: Solução selecionada.
        """
        # Seleciona k índices aleatórios
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        
        # Seleciona o melhor do torneio
        best_idx = indices[0]
        for idx in indices[1:]:
            # Se o atual domina o melhor
            if self.dominates(objectives[idx], objectives[best_idx], population[idx], population[best_idx]):
                best_idx = idx
        
        return population[best_idx]

    def crossover(self, parent1:np.ndarray, parent2:np.ndarray)->tuple:
        """
        Crossover com um ponto de corte aleatório

        Args:
            parent1 (np.ndarray): Primeiro pai.
            parent2 (np.ndarray): Segundo pai.

        Returns:
            tuple: Dois filhos.
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        # Ponto de corte aleatório
        cut_point = random.randint(0, self.num_PAs - 1)
        
        # Crossover
        child1 = np.concatenate([parent1[:cut_point], parent2[cut_point:]])
        child2 = np.concatenate([parent2[:cut_point], parent1[cut_point:]])

        if self.debug:
            # Cria strings visuais para os pais
            parent1_sum = f"{int(sum(parent1[:cut_point])):3d}|{int(sum(parent1[cut_point:])):3d}"
            parent1_dna = '#' * int(cut_point / (self.num_PAs/10))+ '|' + '#' * int((self.num_PAs - cut_point) / (self.num_PAs/10))
            parent2_sum = f"{int(sum(parent2[:cut_point])):3d}|{int(sum(parent2[cut_point:])):3d}"
            parent2_dna = '=' * int(cut_point / (self.num_PAs/10))+ '|' + '=' * int((self.num_PAs - cut_point) / (self.num_PAs/10))
            
            # Cria strings visuais para os filhos
            child1_sum = f"{int(sum(child1[:cut_point])):3d}|{int(sum(child1[cut_point:])):3d}"
            child1_dna = '#' * int(cut_point / (self.num_PAs/10))+ '|' + '=' * int((self.num_PAs - cut_point) / (self.num_PAs/10))
            child2_sum = f"{int(sum(child2[:cut_point])):3d}|{int(sum(child2[cut_point:])):3d}"
            child2_dna = '=' * int(cut_point / (self.num_PAs/10))+ '|' + '#' * int((self.num_PAs - cut_point) / (self.num_PAs/10))
            
            print(f"DEBUG: CROSSOVER - Pais:    [{parent1_dna}]({parent1_sum}) & [{parent2_dna}]({parent2_sum})")
            print(f"DEBUG: CROSSOVER - Filhos:  [{child1_dna}]({child1_sum}) & [{child2_dna}]({child2_sum})")
        
        return child1, child2

    def mutate(self, chromosome:np.ndarray)->np.ndarray:
        """
        Mutação controlada

        Args:
            chromosome (np.ndarray): Cromossomo a ser mutado.

        Returns:
            np.ndarray: Cromossomo mutado.
        """
        chromosome_ = copy.deepcopy(chromosome)
        if random.random() > self.mutation_rate:
            return chromosome_
            
        # Encontra índices de PAs ativos e inativos
        ativos = np.where(chromosome_ == 1)[0]
        inativos = np.where(chromosome_ == 0)[0]
        
        # Escolhe aleatoriamente o tipo de mutação
        tipo_mutacao = random.randint(0, 2)
        
        if tipo_mutacao == 0:  # Adicionar um PA
            if len(inativos) > 0:
                idx = random.choice(inativos)
                chromosome_[idx] = 1
                if self.debug: print(f"DEBUG: MUTACAO - Adicionando PA no índice {idx}")
        
        elif tipo_mutacao == 1:  # Remover um PA
            if len(ativos) > 1:
                idx = random.choice(ativos)
                chromosome_[idx] = 0
                if self.debug: print(f"DEBUG: MUTACAO - Removendo PA no índice {idx}")
        
        else:  # Trocar dois PAs
            if len(ativos) > 0 and len(inativos) > 0:
                idx_ativo = random.choice(ativos)
                idx_inativo = random.choice(inativos)
                chromosome_[idx_ativo] = 0
                chromosome_[idx_inativo] = 1
                if self.debug: print(f"DEBUG: MUTACAO - Trocando índice do PA de {idx_ativo} para {idx_inativo}")
        
        return chromosome_

    def f1_min_distancia(self, users:list):
        """
        A função objetivo f1 é a soma das distâncias entre os usuários e os PAs aos quais eles estão conectados.

        Args:
            users (list): Lista de usuários.

        Returns:
            float: Soma das distâncias entre os usuários e os PAs.
        """
        sum_dist = 0
        for user in users:
            if user.user_atendido:
                sum_dist += np.linalg.norm(np.array(user.coordenadas) - np.array(user.PA_conectado))
        return float(round(sum_dist, 2))

    def f2_min_pas_ativos(self, PAs:list):
        """
        A função objetivo f2 é a soma do número de PAs ativos.

        Args:
            PAs (list): Lista de PAs.

        Returns:
            int: Soma do número de PAs ativos.
        """
        sum_pa = 0
        for pa in PAs:
            if pa.PA_ativado:
                sum_pa += 1
        return sum_pa

    def evaluate(self, chromosome:np.ndarray)->tuple:
        """
        Avalia uma solução (cromossomo).

        Args:
            chromosome (np.ndarray): Vetor binário representando a solução inicial na malha de PAs.

        Returns:
            tuple: Tupla com os valores das funções objetivo.
        """
        # Ativa os PAs de acordo com a solução
        PAs_ativos, users_atendidos = gerar_resultados(chromosome, self.PAs, self.users)
        
        # Calcula f1: soma das distâncias
        f1 = self.f1_min_distancia(users_atendidos)
        
        # Calcula f2: número de PAs ativos
        f2 = self.f2_min_pas_ativos(PAs_ativos)
        
        return (f1, f2)

    def run(self, initial_population:list, save_history:bool=False)->tuple:
        """
        Executa o algoritmo NSGA-II.

        Args:
            initial_population (list): População inicial.
            save_history (bool): Se True, retorna também o histórico de populações e objetivos.

        Returns:
            tuple: Tupla com a fronteira de Pareto, seus valores objetivo, 
            o histórico de populações e o histórico de objetivos das populações (se save_history=True).
        """        
        # Avalia a população inicial
        population = initial_population
        self.pop_size = len(population)
        objectives = [self.evaluate(sol) for sol in population]
        
        # Inicializa histórico se necessário
        if save_history:
            history_populations = []
            history_objectives = []
        
        # Armazena o melhor hipervolume encontrado
        best_hypervolume = 0
        best_pareto_front = []
        
        for generation in range(self.max_gen):
            # Limpa o cache de violação de restrições para a nova geração
            self.clear_constraint_cache()
            
            if self.verbose or self.debug: print(f"\n=== GERAÇÃO {generation+1} ===")

            # Checa violação de restrições de pais
            if self.debug: print(f"\nDEBUG: Iniciando checagem de violação de restrições dos pais")
            for i, sol in enumerate(population):
                self.check_constraints(sol, objectives[i])

            # 1. Seleção, cruzamento e mutação
            if self.debug: print(f"\nDEBUG: Iniciando seleção, cruzamento e mutação")
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection(population, objectives)
                p2 = self.tournament_selection(population, objectives)
                child1, child2 = self.crossover(p1, p2)
                offspring.extend([self.mutate(child1), self.mutate(child2)])
            if self.verbose: print(f"{len(offspring)} filhos gerados")
            objectives_offspring = [self.evaluate(sol) for sol in offspring]
            
            # Checa violação de restrições de filhos
            if self.debug: print(f"\nDEBUG: Iniciando checagem de violação de restrições dos filhos")
            for i, sol in enumerate(offspring):
                self.check_constraints(sol, objectives_offspring[i])
                
            # 2. Combina população pai e filhos
            combined_population = population + offspring
            combined_objectives = objectives + objectives_offspring
            if self.verbose: print(f"{len(combined_population)} individuos na população combinada: {combined_objectives}")
            
            # 3. Ordenação não dominada da população combinada
            if self.debug: print(f"\nDEBUG: Iniciando ordenação não dominada")
            fronts = self.non_dominated_sort(combined_population, combined_objectives)
            if self.verbose: print(f"{len(fronts)} frentes encontradas: {fronts}")
            if len(fronts) > 0:
                # Mantém as soluções diretamente na frente de Pareto
                pareto_solutions = [combined_population[idx] for idx in fronts[0]]
            else:
                pareto_solutions = []
            
            # 4. Calcula distância de aglomeração para cada frente
            if self.debug: print(f"\nDEBUG: Iniciando cálculo de distância de aglomeração")
            crowding_distances = {}
            for i, front in enumerate(fronts):
                if self.debug: print(f"DEBUG: AGLOMERACAO - Avaliando fronteira {i}: {front}")
                if len(front) > 0:  # Verifica se a frente não está vazia
                    if len(front) == 1:
                        # Para frentes com uma única solução, atribui distância muito alta
                        crowding_distances[i] = [np.inf] * len(front)
                        if self.debug: print(f"DEBUG: AGLOMERACAO - Frente {i} tem uma única solução. Distância de aglomeração: infinita")
                    else:
                        # Obtém os valores objetivos da frente atual usando os índices
                        crowding_distances[i] = self.crowding_distance(front, combined_objectives)
                        if self.debug: print(f"DEBUG: AGLOMERACAO - Distância de aglomeração calculada para frente {i} com {len(front)} soluções")
            if self.verbose: print(f"Distância de aglomeração calculada para {len([f for f in fronts if len(f) > 0])} frentes não vazias")
            
            # 5. Cria nova população de tamanho N
            new_population = []
            new_objectives = []
            
            if self.debug: print(f"\nDEBUG: Iniciando criação de nova população")
            for i, front in enumerate(fronts):
                # Se adicionar esta frente exceder o tamanho da população
                if len(new_population) + len(front) > self.pop_size:
                    # Ordena a frente atual por distância de aglomeração
                    sorted_front = sorted(zip(front, crowding_distances[i]), key=lambda x: -x[1])
                    # Adiciona apenas o número necessário de soluções
                    needed = self.pop_size - len(new_population)
                    for j in range(needed):
                        idx = sorted_front[j][0]
                        new_population.append(combined_population[idx])
                        new_objectives.append(combined_objectives[idx])
                    
                    if self.debug: print(f"DEBUG: NOVA POPULACAO - Adicionadas {needed} soluções da frente {i}")
                    break
                # Se não exceder, adiciona toda a frente
                else:
                    for idx in front:
                        new_population.append(combined_population[idx])
                        new_objectives.append(combined_objectives[idx])
                    
                    if self.debug: print(f"DEBUG: NOVA POPULACAO - Adicionadas {len(front)} soluções da frente {i}")
            
            # Atualiza população e objetivos
            population = new_population
            objectives = new_objectives
            
            # Encontra os índices das soluções da frente de Pareto na nova população
            pareto_front = []
            for i, sol in enumerate(population):
                if any(np.array_equal(sol, ps) for ps in pareto_solutions):
                    pareto_front.append(i)
            
            if self.verbose: print(f"População atualizada com {len(population)} soluções: {objectives}")
            
            # Calcula hipervolume da fronteira atual
            if len(pareto_front) > 0:
                hv = self.calculate_hypervolume([objectives[i] for i in pareto_front])
                if self.verbose: print(f"Hipervolume da fronteira atual: {round(hv, 2)}")
            else:
                hv = 0
                if self.verbose: print("Não há soluções na fronteira de Pareto")
            
            # Atualiza melhor hipervolume
            if hv > best_hypervolume:
                if self.verbose: print(f"Novo melhor hipervolume encontrado!")
                best_hypervolume = hv
                best_pareto_front = [population[i] for i in pareto_front]

            if self.debug: print(f"DEBUG: CACHE - {self.constraint_violation_cache.values()}")
            
            # Salva a última população se necessário
            if save_history:
                history_populations.append(population)
                history_objectives.append(objectives)
            
        # Retorna as soluções da melhor frente de Pareto e seus objetivos
        if save_history:
            return best_pareto_front, [self.evaluate(sol) for sol in best_pareto_front], history_populations, history_objectives
        else:
            return best_pareto_front, [self.evaluate(sol) for sol in best_pareto_front], [], []

    def visualize_population_fronts(self, population:list, objectives:list, titulo:str=None, x_limits:tuple=None, y_limits:tuple=None, save_fig:bool=False, fig_name:str=None):
        """
        Visualiza todas as fronteiras da população atual.
        
        Args:
            population (list): Lista de soluções.
            population (list): Lista de soluções.
            objectives (list): Lista de tuplas com resultados das funções objetivo.
            titulo (str): Título do gráfico.
            x_limits (tuple): Limites do eixo x.
            y_limits (tuple): Limites do eixo y.
            save_fig (bool): Se True, salva o gráfico.
            fig_name (str): Nome do arquivo do gráfico.
        """
        # Ordena a população em frentes
        fronts = self.non_dominated_sort(population, objectives)
        
        # Cores para cada frente (usando o colormap viridis)
        colors = plt.cm.viridis(np.linspace(0, 1, len(fronts)))
        
        # Para cada frente
        for i, front in enumerate(fronts):
            # Obtém os valores objetivos da frente
            front_objectives = [objectives[idx] for idx in front]
            f1_vals, f2_vals = zip(*front_objectives)
            
            # Ordena os pontos para linhas suaves
            sorted_indices = np.argsort(f1_vals)
            f1_vals = np.array(f1_vals)[sorted_indices]
            f2_vals = np.array(f2_vals)[sorted_indices]
            
            # Plota os pontos e as linhas
            plt.scatter(f1_vals, f2_vals, color=colors[i], label=f'Fronteira {i+1} ({len(front)})')
            plt.plot(f1_vals, f2_vals, color=colors[i], alpha=0.5)

            # Limita os eixos
            if x_limits:
                plt.xlim(x_limits)
            if y_limits:
                plt.ylim(y_limits)
        
        plt.xlabel("Soma das Distâncias (m)")
        plt.ylabel("Número de PAs Ativos")
        if titulo != "" and titulo != None:
            plt.title(titulo)
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(f"{fig_name}.png", dpi=300)
        plt.show()

    def visualize_results(self, pareto_solutions:list, pareto_objectives:list, titulo:str=None, excluir_infactiveis:bool=False):
        """
        Visualiza a fronteira de Pareto e calcula métricas

        Args:
            pareto_solutions (list): Lista de soluções na fronteira de Pareto.
            pareto_objectives (list): Lista de tuplas com resultados das funções objetivo.
            titulo (str): Título do gráfico.
            excluir_infactiveis (bool): Se True, exclui soluções infactíveis do plot.

        Returns:
            tuple: Hipervolume final e fronteira de Pareto.
        """
        # Verifica factibilidade de cada solução
        factiveis = []
        infactiveis = []
        f1_factiveis = []
        f2_factiveis = []
        f1_infactiveis = []
        f2_infactiveis = []
        
        for sol, obj in zip(pareto_solutions, pareto_objectives):
            if self.check_constraints(sol, obj):
                factiveis.append(sol)
                f1_factiveis.append(obj[0])
                f2_factiveis.append(obj[1])
            else:
                infactiveis.append(sol)
                f1_infactiveis.append(obj[0])
                f2_infactiveis.append(obj[1])

        unicas = []
        for sol in pareto_solutions:
            sol_tuple = tuple(sol)
            if sol_tuple not in unicas:
                unicas.append(sol_tuple)
        unicas = [np.array(sol) for sol in unicas]

        print(f"Soluções Factíveis:  {len(factiveis)}")
        print(f"Soluções Infacíveis: {len(infactiveis)}")
        print(f"Soluções Únicas:     {len(unicas)}")

        # Calcula hipervolume apenas das soluções factíveis
        if excluir_infactiveis:
            hv = self.calculate_hypervolume(list(zip(f1_factiveis, f2_factiveis)))
            print(f"Hipervolume da fronteira (apenas factíveis): {hv}")
        else:
            hv = self.calculate_hypervolume(pareto_objectives)
            print(f"Hipervolume da fronteira: {hv}")

        # Plota as soluções factíveis
        plt.scatter(f1_factiveis, f2_factiveis, color='red', label='Soluções Factíveis')
        
        # Plota as soluções infactíveis
        if not excluir_infactiveis and len(infactiveis) > 0:
            plt.scatter(f1_infactiveis, f2_infactiveis, color='black', marker='x', label='Soluções Infactíveis')
            plt.legend()
        plt.xlabel("Soma das Distâncias (m)")
        plt.ylabel("Número de PAs Ativos")
        if titulo != "" and titulo != None:
            plt.title(titulo)
        plt.grid(True)
        plt.show()