# --- 1. Importações: Trazendo as "Caixas de Ferramentas" ---
# Cada 'import' traz uma biblioteca que nos dá superpoderes para realizar tarefas específicas.

from flask import Flask, render_template, request, url_for  # Flask: Para criar a aplicação web (rotas, templates).
from ged4py.parser import GedcomReader                  # Ged4py: Para ler e interpretar o arquivo GEDCOM.
from thefuzz import fuzz                                # TheFuzz: Para comparar nomes de forma flexível (ex: "José" e "Jose").
import pandas as pd                                     # Pandas: A melhor ferramenta para trabalhar com dados de tabelas (nosso CSV).
import itertools                                        # Itertools: Ferramentas de iteração, usamos para limitar o número de caminhos.
import networkx as nx                                   # NetworkX: Para criar, manipular e analisar grafos (nossa árvore genealógica).
from pyvis.network import Network                       # Pyvis: Para criar as visualizações de grafos interativas em HTML.
import os                                               # OS: Para interagir com o sistema operacional (criar pastas, manipular caminhos).

# --- 2. Configuração Inicial da Aplicação ---

# Cria a instância principal da nossa aplicação Flask.
app = Flask(__name__)
# Define uma "chave secreta" para o Flask. É necessário para algumas funcionalidades de segurança.
app.secret_key = 'f@milyse@rch_dna_edition_v6'

# --- 3. Variáveis Globais: A "Memória" da Nossa Aplicação ---
# Estas variáveis vão guardar os dados carregados (pessoas e o grafo) enquanto a aplicação estiver rodando.
# Em aplicações maiores, isso seria feito com um banco de dados.
people = {}
graph = None

# --- 4. Dados de Referência: Nosso "Minidicionário" Genealógico ---
# Esta lista é nossa base de conhecimento para traduzir valores de cM em relacionamentos prováveis.
# Baseado no "Shared cM Project".
SHARED_CM_DATA = [
    {"range": (2376, 3950), "relationship": "Pais/Filhos, Irmãos completos"},
    {"range": (1317, 2312), "relationship": "Avós/Netos, Tios/Sobrinhos, Meios-irmãos"},
    {"range": (575, 1330), "relationship": "Primos de 1º grau, Tios-avós/Sobrinhos-netos"},
    {"range": (215, 728), "relationship": "Primos de 1º grau (1x removido), Meios-primos"},
    {"range": (73, 504), "relationship": "Primos de 2º grau"},
    {"range": (30, 217), "relationship": "Primos de 2º grau (1x removido)"},
    {"range": (0, 147), "relationship": "Primos de 3º grau"},
    {"range": (0, 89), "relationship": "Primos de 3º grau (1x removido)"},
    {"range": (0, 71), "relationship": "Primos de 4º grau"},
]

# --------------------------------------------------------------------
# --- 5. Funções Auxiliares: Os "Assistentes" da Nossa Aplicação ---
# Dividimos o código em funções para que cada uma tenha uma única responsabilidade.
# Isso torna o código mais organizado, legível e fácil de consertar.
# --------------------------------------------------------------------

def get_name(person):
    """Uma pequena função para extrair o nome de um objeto 'pessoa' do GEDCOM de forma segura."""
    return person.name.format() if person.name else "Sem Nome"

def get_relationships_by_cm(cm_value):
    """
    Recebe um valor de cM e consulta nossa lista SHARED_CM_DATA.
    Retorna uma lista de todos os relacionamentos possíveis para aquele valor.
    """
    if not isinstance(cm_value, (int, float)) or cm_value <= 0:
        return []
    
    # Procura em nossa lista de referência quais faixas contêm o valor de cM.
    possible_relationships = [item["relationship"] for item in SHARED_CM_DATA if item["range"][0] <= cm_value <= item["range"][1]]
    
    # Se nenhuma faixa corresponder, retorna uma mensagem padrão.
    return possible_relationships if possible_relationships else ["Relação distante ou indeterminada"]

def load_gedcom_and_build_graph(file_path):
    """Função principal que orquestra a leitura do GEDCOM e a construção do grafo."""
    global people, graph  # Avisa ao Python que vamos modificar as variáveis globais.
    with GedcomReader(file_path) as parser:
        # Cria um dicionário onde a chave é o ID da pessoa (ex: '@I1@') e o valor é o objeto pessoa.
        people = {i.xref_id: i for i in parser.records0("INDI")}
        # Chama a função que constrói o grafo.
        graph = build_graph_from_parser(people, parser)

def build_graph_from_parser(people_dict, parser):
    """
    Esta é uma das partes mais importantes. Ela cria o grafo NetworkX.
    O modelo usado é "bipartido": Pessoas se conectam a Famílias, e Famílias se conectam a outras Pessoas.
    Isso modela corretamente as relações de pais e filhos.
    """
    g = nx.Graph()
    # Adiciona um "nó" no grafo para cada pessoa.
    for pid, person in people_dict.items():
        g.add_node(pid, label=get_name(person), type='person')
    # Adiciona um "nó" no grafo para cada família.
    for fam in parser.records0("FAM"):
        if fam.xref_id: g.add_node(fam.xref_id, label='Familia', type='family')
    
    # Agora, cria as "arestas" (linhas) que conectam as pessoas às suas famílias.
    for fam in parser.records0("FAM"):
        fam_id = fam.xref_id
        if not fam_id: continue
        
        # Encontra o marido, esposa e filhos dentro do registro da família.
        husband_id, wife_id, child_ids = None, None, []
        for sub_rec in fam.sub_records:
            if sub_rec.tag == "HUSB": husband_id = sub_rec.value
            elif sub_rec.tag == "WIFE": wife_id = sub_rec.value
            elif sub_rec.tag == "CHIL": child_ids.append(sub_rec.value)
        
        # Cria as conexões (arestas) no grafo.
        if husband_id and g.has_node(husband_id): g.add_edge(husband_id, fam_id, weight=1)
        if wife_id and g.has_node(wife_id): g.add_edge(wife_id, fam_id, weight=1)
        for child_id in child_ids:
            if g.has_node(child_id): g.add_edge(child_id, fam_id, weight=1)
    return g

def find_person_by_name(name_query):
    """Procura em nosso dicionário 'people' por uma pessoa com um nome correspondente."""
    matches = [pid for pid, p in people.items() if name_query.lower() in get_name(p).lower()]
    return matches

def find_connections(start, end, limit=1):
    """Usa o poder do NetworkX para encontrar o caminho genealógico mais curto entre duas pessoas."""
    try:
        # nx.shortest_simple_paths é o algoritmo que faz a mágica de encontrar o caminho.
        paths_generator = nx.shortest_simple_paths(graph, source=start, target=end, weight='weight')
        # Retorna apenas o primeiro caminho encontrado (o mais curto).
        return list(itertools.islice(paths_generator, limit))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Se não houver um caminho, retorna uma lista vazia.
        return []

def plot_connection_pyvis(connection_path, start_node, end_node, output_filename):
    """
    Esta função é a responsável por criar a visualização bonita e interativa.
    Ela recebe um caminho (uma lista de IDs de pessoas e famílias) e desenha o grafo correspondente.
    """
    # Cria uma instância da rede Pyvis que será desenhada.
    vis_net = Network(height='400px', width='100%', directed=True, notebook=False, cdn_resources='remote')
    
    # Extrai apenas os nós de pessoas do caminho completo para uma visualização mais limpa.
    person_nodes_in_path = [node for node in connection_path if graph.nodes[node].get('type') == 'person']

    # Adiciona cada pessoa como um nó no grafo visual, com cores e tamanhos diferentes.
    for node_id in person_nodes_in_path:
        person = people[node_id]
        label = get_name(person)
        color, size = 'lightblue', 15
        if node_id == start_node: color, size = '#ff4747', 25  # Vermelho para o usuário
        elif node_id == end_node: color, size = '#47ff75', 25 # Verde para o match
        vis_net.add_node(node_id, label=label, title=label, color=color, size=size)
        
    # Adiciona as arestas (linhas) conectando as pessoas em sequência.
    for i in range(len(person_nodes_in_path) - 1):
        source_node, target_node = person_nodes_in_path[i], person_nodes_in_path[i+1]
        vis_net.add_edge(source_node, target_node)
        
    # Configura o layout para ser hierárquico (em árvore), da esquerda para a direita.
    vis_net.set_options("""
    var options = {"layout": {"hierarchical": {"enabled": true, "direction": "LR", "sortMethod": "directed", "levelSeparation": 150, "nodeSpacing": 120}},"physics": {"enabled": false}}
    """)
    
    # Salva o grafo gerado como um arquivo HTML na pasta 'static'.
    filepath = os.path.join('static', output_filename)
    vis_net.save_graph(filepath)
    return output_filename # Retorna o nome do arquivo para ser usado no HTML.

# --------------------------------------------------------------------
# --- 6. Rota Principal: O "Cérebro" da Aplicação Web ---
# Esta função é executada toda vez que alguém acessa a página principal.
# --------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # Se o método for POST, significa que o usuário preencheu e enviou o formulário.
    if request.method == "POST":
        
        # --- Parte A: Validação dos Dados de Entrada ---
        if "gedcom" not in request.files or "matches_csv" not in request.files or "root_name" not in request.form:
            return render_template("index.html", message="Todos os campos são obrigatórios.", success=False)
        
        gedcom_file, matches_file, root_name = request.files["gedcom"], request.files["matches_csv"], request.form["root_name"]
        
        if gedcom_file.filename == '' or matches_file.filename == '' or root_name == '':
            return render_template("index.html", message="Por favor, preencha todos os campos.", success=False)
        
        try:
            # --- Parte B: Processamento dos Arquivos ---
            # Garante que as pastas para salvar os arquivos existem.
            os.makedirs("uploads", exist_ok=True); os.makedirs("static", exist_ok=True)
            # Salva os arquivos no servidor para poderem ser lidos.
            gedcom_path, matches_path = os.path.join("uploads", gedcom_file.filename), os.path.join("uploads", matches_file.filename)
            gedcom_file.save(gedcom_path); matches_file.save(matches_path)
            
            # Chama nossas funções auxiliares para carregar os dados.
            load_gedcom_and_build_graph(gedcom_path)
            dna_matches_df = pd.read_csv(matches_path, encoding='utf-8', skipinitialspace=True)
            
            # Limpa os nomes das colunas do CSV (remove espaços extras).
            dna_matches_df.columns = [col.strip() for col in dna_matches_df.columns]
            
            # Encontra o ID do usuário (pessoa raiz) na árvore.
            root_person_ids = find_person_by_name(root_name)
            if not root_person_ids:
                return render_template("index.html", message=f"Seu nome '{root_name}' não foi encontrado no arquivo GEDCOM.", success=False)
            root_id = root_person_ids[0]
            
            # --- Parte C: Lógica de Agrupamento e Análise ---
            # Detecta os nomes corretos das colunas de "Nome" e "cM" no CSV.
            possible_name_columns = ['Name', 'MatchedName', 'Nome']
            possible_cm_columns = ['cM', 'TotalCM', 'Total cM']
            name_col = next((col for col in possible_name_columns if col in dna_matches_df.columns), None)
            cm_col = next((col for col in possible_cm_columns if col in dna_matches_df.columns), None)

            if not name_col or not cm_col:
                return render_template("index.html", message=f"Não foi possível encontrar as colunas de Nome e/ou cM no CSV.", success=False)
            
            # Agrupa os segmentos de DNA por pessoa e soma os cM para obter o total.
            aggregated_matches = dna_matches_df.groupby(name_col).agg({cm_col: 'sum'}).reset_index()
            aggregated_matches.rename(columns={cm_col: 'TotalCM_Calculated'}, inplace=True)
            
            # --- Parte D: Geração dos Resultados ---
            results_list = []
            NOME_MATCH_MIN_SIMILARITY = 92 # Limiar de similaridade para nomes.
            
            # Limpa os grafos da análise anterior.
            for f in os.listdir('static'):
                if f.startswith('graph_') and f.endswith('.html'): os.remove(os.path.join('static', f))
            
            match_counter = 0
            # Itera sobre a lista de matches JÁ AGRUPADOS.
            for index, row in aggregated_matches.iterrows():
                match_name = row[name_col]
                # Itera sobre as pessoas do GEDCOM para encontrar uma correspondência.
                for pid, person in people.items():
                    gedcom_name = get_name(person)
                    similarity = fuzz.ratio(str(match_name).lower(), gedcom_name.lower())
                    
                    if similarity >= NOME_MATCH_MIN_SIMILARITY:
                        connections = find_connections(root_id, pid) # Encontra o caminho
                        if connections:
                            path = connections[0]
                            person_nodes_for_text = [node for node in path if graph.nodes[node]['type'] == 'person']
                            nomes = [get_name(people[p_id]) for p_id in person_nodes_for_text]
                            
                            cm_value = row.get('TotalCM_Calculated', 0)
                            probable_relationships = get_relationships_by_cm(cm_value) # Prevê o parentesco
                            
                            graph_filename = f"graph_{match_counter}.html"
                            plot_connection_pyvis(path, start_node=root_id, end_node=pid, output_filename=graph_filename) # Desenha o grafo
                            
                            # Monta o dicionário de resultado para este match.
                            results_list.append({
                                'match_name': gedcom_name, 'cm': cm_value, 'text_path': " → ".join(nomes),
                                'graph_file': graph_filename, 'relationships': ", ".join(probable_relationships)
                            })
                            match_counter += 1
                        break # Pára de procurar no GEDCOM assim que encontrar o match.
            
            # Ordena a lista de resultados final pelo maior valor de cM.
            results_list_sorted = sorted(results_list, key=lambda x: x.get('cm', 0), reverse=True)
            
            # Envia os resultados para a página HTML para serem exibidos.
            return render_template("index.html", has_results=True, results_list=results_list_sorted, message=f"{len(results_list_sorted)} conexões encontradas.", success=True)
        
        except Exception as e:
            # Se algo der errado no meio do caminho, mostra uma mensagem de erro clara.
            return render_template("index.html", message=f"Ocorreu um erro durante o processamento: {e}", success=False)
            
    # Se o método for GET (o usuário apenas abriu a página), simplesmente mostra a página vazia.
    return render_template("index.html")

# --------------------------------------------------------------------
# --- 7. Inicialização da Aplicação ---
# --------------------------------------------------------------------

# Este bloco só é executado quando você roda o script diretamente (ex: 'python app.py').
if __name__ == "__main__":
    # Inicia o servidor de desenvolvimento do Flask.
    # debug=True é ótimo para desenvolver, pois atualiza o servidor automaticamente
    # quando você salva o arquivo e mostra erros detalhados no navegador.
    app.run(debug=True)