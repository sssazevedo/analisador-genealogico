import os
import re
from flask import Flask, render_template, request, url_for
from ged4py.parser import GedcomReader
from thefuzz import fuzz
import pandas as pd
import itertools
import unicodedata
import networkx as nx
from collections import deque

# --- Configuração ---
app = Flask(__name__)
app.secret_key = 'f@milyse@rch_dna_edition_v16'

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

people = {}
families = {}
graph = None
child_to_family = {}  # <-- índice filho -> [famílias onde é CHIL]

SHARED_CM_DATA = [
    {"range": (2376, 3950), "relationship": "Pais/Filhos, Irmãos completos"},
    {"range": (1317, 2312), "relationship": "Avós/Netos, Tios/Sobrinhos, Meios-irmãos"},
    {"range": (575, 1330), "relationship": "Primos de 1º grau, Tios-avós/Sobrinhos-netos"},
    {"range": (215, 728), "relationship": "Primos de 1º grau (1x removido), Meios-primos"},
    {"range": (73, 504), "relationship": "Primos de 2º grau"},
]

# ---------- Helpers ----------
def ref_id(val):
    """Retorna o xref ('@I..@' / '@F..@') de um ponteiro do ged4py ou a própria string."""
    return getattr(val, 'xref_id', val)

# --- Funções Auxiliares ---
def get_name(person):
    return person.name.format() if person and person.name else "Sem Nome"

def get_relationships_by_cm(cm_value):
    if not isinstance(cm_value, (int, float)) or cm_value <= 0:
        return []
    possible = [item["relationship"] for item in SHARED_CM_DATA if item["range"][0] <= cm_value <= item["range"][1]]
    return possible if possible else ["Relação distante ou indeterminada"]

def load_gedcom_and_build_graph(file_path):
    """Carrega o GEDCOM (encoding auto) e constrói o grafo + índices auxiliares."""
    global people, families, graph, child_to_family
    with GedcomReader(file_path) as parser:  # deixa auto-detectar pelo HEAD/CHAR
        people   = {ref_id(i.xref_id): i for i in parser.records0("INDI")}
        families = {ref_id(f.xref_id): f for f in parser.records0("FAM")}
        graph, child_to_family = build_graph_from_parser(people, parser)
        all_names = sorted([get_name(p) for p in people.values()])
        return all_names

def build_graph_from_parser(people_dict, parser):
    """Gera grafo Pessoa <-> Família e índice child_to_family (filho -> [famílias])."""
    g = nx.Graph()
    c2f = {}

    # Pessoas
    for pid, person in people_dict.items():
        g.add_node(pid, label=get_name(person), type='person')

    # Famílias + ligações
    for fam in parser.records0("FAM"):
        fam_id = ref_id(fam.xref_id)
        if not fam_id:
            continue
        g.add_node(fam_id, label='Familia', type='family')

        husband_id, wife_id = None, None
        child_ids = []
        for sub_rec in fam.sub_records:
            if sub_rec.tag == "HUSB":
                husband_id = ref_id(sub_rec.value)
            elif sub_rec.tag == "WIFE":
                wife_id = ref_id(sub_rec.value)
            elif sub_rec.tag == "CHIL":
                cid = ref_id(sub_rec.value)
                child_ids.append(cid)
                c2f.setdefault(cid, []).append(fam_id)  # <-- índice filho->família

        if husband_id and g.has_node(husband_id):
            g.add_edge(husband_id, fam_id)
        if wife_id and g.has_node(wife_id):
            g.add_edge(wife_id, fam_id)
        for cid in child_ids:
            if g.has_node(cid):
                g.add_edge(cid, fam_id)

    return g, c2f

def find_person_by_name(name_query):
    exact = [pid for pid, p in people.items() if name_query.lower() == get_name(p).lower()]
    if exact:
        return exact
    return [pid for pid, p in people.items() if name_query.lower() in get_name(p).lower()]

def get_parents(person_id):
    """Retorna pais. 1º tenta FAMC; se não houver, usa o índice de famílias onde a pessoa é CHIL."""
    person = people.get(person_id)
    if not person:
        return []

    # 1) Caminho “normal”: FAMC no indivíduo
    famc_ref = next((ref_id(rec.value) for rec in person.sub_records if rec.tag == "FAMC"), None)
    fam_ids = []
    if famc_ref:
        fam_ids = [famc_ref]
    else:
        # 2) Fallback (caso deste GEDCOM): procurar famílias onde a pessoa é CHIL
        fam_ids = child_to_family.get(person_id, [])

    if not fam_ids:
        return []

    parent_ids = []
    for fam_id in fam_ids:
        family = families.get(fam_id)
        if not family:
            continue
        for sub_rec in family.sub_records:
            if sub_rec.tag in ("HUSB", "WIFE"):
                pid = ref_id(sub_rec.value)
                if pid and pid not in parent_ids:
                    parent_ids.append(pid)
    return parent_ids

def get_spouses(person_id):
    person = people.get(person_id)
    spouse_ids = []

    # 1) Caminho normal: via FAMS no indivíduo
    if person:
        fams_refs = [ref_id(rec.value) for rec in person.sub_records if rec.tag == "FAMS"]
        for fam_ref in fams_refs:
            family = families.get(fam_ref)
            if not family:
                continue
            is_husband = any(ref_id(rec.value) == person_id for rec in family.sub_records if rec.tag == "HUSB")
            partner_tag = "WIFE" if is_husband else "HUSB"
            for rec in family.sub_records:
                if rec.tag == partner_tag:
                    spouse_ids.append(ref_id(rec.value))

    if spouse_ids:
        return spouse_ids

    # 2) Fallback: varre FAMs onde a pessoa aparece como HUSB/WIFE (mesmo sem FAMS no INDI)
    for fam in families.values():
        husb = next((ref_id(r.value) for r in fam.sub_records if r.tag == "HUSB"), None)
        wife = next((ref_id(r.value) for r in fam.sub_records if r.tag == "WIFE"), None)
        if husb == person_id and wife:
            spouse_ids.append(wife)
        elif wife == person_id and husb:
            spouse_ids.append(husb)

    return spouse_ids


def find_ancestral_path(start_id, end_id, max_depth=20):
    """BFS bilateral: sobe somente pelos pais (mantém sua regra)."""
    q1, q2 = deque([(start_id, [start_id])]), deque([(end_id, [end_id])])
    visited1, visited2 = {start_id: [start_id]}, {end_id: [end_id]}
    if start_id == end_id:
        return ([start_id], start_id)

    for _depth in range(max_depth):
        q_size = len(q1)
        if not q_size:
            break
        for _ in range(q_size):
            curr_id, path = q1.popleft()
            if curr_id in visited2:
                return (path + visited2[curr_id][::-1][1:], curr_id)
            for p_id in get_parents(curr_id):
                if p_id not in visited1:
                    new_path = path + [p_id]
                    visited1[p_id] = new_path
                    q1.append((p_id, new_path))

        q_size = len(q2)
        if not q_size:
            break
        for _ in range(q_size):
            curr_id, path = q2.popleft()
            if curr_id in visited1:
                return (visited1[curr_id] + path[::-1][1:], curr_id)
            for p_id in get_parents(curr_id):
                if p_id not in visited2:
                    new_path = path + [p_id]
                    visited2[p_id] = new_path
                    q2.append((p_id, new_path))
    return (None, None)

# --- INÍCIO DA CORREÇÃO DEFINITIVA ---
def generate_mermaid_graph(path, p1_id, p2_id, common_ancestor_id):
    """Gera um gráfico Mermaid.js robusto em formato "V" (BT - Bottom Top) ou linear."""
    import re

    def sid(raw: str) -> str:
        safe_str = str(raw).replace('@', '').replace('+', '_')
        return 'N_' + re.sub(r'[^a-zA-Z0-9_]', '', safe_str)

    def lab(txt: str) -> str:
        s = unicodedata.normalize("NFC", str(txt))
        s = (s.replace('\u00A0', ' ')
               .replace('\u2013', '-')
               .replace('\u2014', '-')
               .replace('“','"').replace('”','"').replace('’', "'"))
        s = (s.replace('&', '&amp;')
               .replace('<', '&lt;')
               .replace('>', '&gt;'))
        s = (s.replace('\\', r'\\')
               .replace('"', r'\"')
               .replace('[', r'\[').replace(']', r'\]')
               .replace('{', r'\{').replace('}', r'\}'))
        s = re.sub(r'[\r\n]+', ' ', s)
        return s

    lines = ["flowchart BT"]; seen_nodes = set()

    ac_idx = -1
    if common_ancestor_id and common_ancestor_id in path:
        try:
            ac_idx = path.index(common_ancestor_id)
        except ValueError:
            pass

    is_direct_ancestry = (ac_idx == -1 or ac_idx == 0 or ac_idx == len(path) - 1)

    couple_members_to_skip = set()
    ancestor_id_for_arrows = sid(common_ancestor_id)

    if not is_direct_ancestry:
        spouses = get_spouses(common_ancestor_id)
        if spouses:
            couple_members_to_skip.update({common_ancestor_id, spouses[0]})
            couple_id_str = "+".join(sorted(list(couple_members_to_skip)))
            ancestor_id_for_arrows = sid(couple_id_str)
            ac_name1 = lab(get_name(people[common_ancestor_id]))
            ac_name2 = lab(get_name(people[spouses[0]]))
            lines.append(f'{ancestor_id_for_arrows}["{ac_name1} &amp; {ac_name2}"]')
            seen_nodes.add(ancestor_id_for_arrows)

    for node_id in path:
        if node_id in couple_members_to_skip:
            continue
        node_sid = sid(node_id)
        if node_sid not in seen_nodes:
            node_name = lab(get_name(people.get(node_id)))
            lines.append(f'{node_sid}["{node_name}"]')
            seen_nodes.add(node_sid)

    if is_direct_ancestry:
        for i in range(len(path) - 1):
            lines.append(f'{sid(path[i])} --> {sid(path[i+1])}')
    elif ac_idx > 0:
        for i in range(ac_idx - 1):
            lines.append(f'{sid(path[i])} --> {sid(path[i+1])}')
        lines.append(f'{sid(path[ac_idx - 1])} --> {ancestor_id_for_arrows}')
        for i in range(len(path) - 1, ac_idx + 1, -1):
            lines.append(f'{sid(path[i])} --> {sid(path[i-1])}')
        lines.append(f'{sid(path[ac_idx + 1])} --> {ancestor_id_for_arrows}')

    lines.append(f'style {sid(p1_id)} fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px')
    lines.append(f'style {sid(p2_id)} fill:#ffebee,stroke:#ef5350,stroke-width:2px')
    if common_ancestor_id:
        lines.append(f'style {ancestor_id_for_arrows} fill:#fff9c4,stroke:#fbc02d,stroke-width:2px')

    return "\n".join(lines)
# --- FIM DA CORREÇÃO DEFINITIVA ---


# --- Rota Principal ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload_gedcom":
            if "gedcom" not in request.files:
                return render_template("index.html", message="Nenhum arquivo GEDCOM enviado.", success=False)
            gedcom_file = request.files["gedcom"]
            if gedcom_file.filename == '':
                return render_template("index.html", message="Nenhum arquivo selecionado.", success=False)
            try:
                gedcom_path = os.path.join(UPLOAD_FOLDER, gedcom_file.filename)
                gedcom_file.save(gedcom_path)
                all_names = load_gedcom_and_build_graph(gedcom_path)
                return render_template("index.html", gedcom_filename=gedcom_file.filename, all_names=all_names, message=f"Arquivo '{gedcom_file.filename}' carregado!", success=True)
            except Exception as e:
                return render_template("index.html", message=f"Erro ao processar GEDCOM: {e}", success=False)

        gedcom_filename = request.form.get("gedcom_filename")
        if not gedcom_filename:
            return render_template("index.html", message="Erro: Arquivo GEDCOM não encontrado.", success=False)
        gedcom_path = os.path.join(UPLOAD_FOLDER, gedcom_filename)
        if not os.path.exists(gedcom_path):
            return render_template("index.html", message=f"Erro: Arquivo '{gedcom_filename}' não existe mais.", success=False)
        all_names = load_gedcom_and_build_graph(gedcom_path)

        if action == "dna_analysis":
            try:
                if "matches_csv" not in request.files or not request.files["matches_csv"].filename:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message="Por favor, carregue o arquivo CSV de matches.", success=False)
                matches_file, root_name = request.files["matches_csv"], request.form["root_name"]
                matches_path = os.path.join(UPLOAD_FOLDER, matches_file.filename)
                matches_file.save(matches_path)
                dna_matches_df = pd.read_csv(matches_path, encoding='utf-8', skipinitialspace=True)
                dna_matches_df.columns = [col.strip() for col in dna_matches_df.columns]
                root_person_ids = find_person_by_name(root_name)
                if not root_person_ids:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Seu nome '{root_name}' não foi encontrado no GEDCOM.", success=False)
                root_id = root_person_ids[0]
                name_col = next((col for col in ['Name', 'MatchedName', 'Nome'] if col in dna_matches_df.columns), None)
                cm_col = next((col for col in ['cM', 'TotalCM', 'Total cM'] if col in dna_matches_df.columns), None)
                if not name_col or not cm_col:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message="Colunas de Nome e/ou cM não encontradas.", success=False)
                aggregated_matches = dna_matches_df.groupby(name_col).agg({cm_col: 'sum'}).reset_index()
                results_list = []
                NOME_MATCH_MIN_SIMILARITY = 80
                for _, row in aggregated_matches.iterrows():
                    match_name = str(row[name_col])
                    for pid, person in people.items():
                        gedcom_name = get_name(person)
                        if fuzz.ratio(match_name.lower(), gedcom_name.lower()) >= NOME_MATCH_MIN_SIMILARITY:
                            path, common_ancestor = find_ancestral_path(root_id, pid)
                            if path:
                                nomes = [get_name(people[p_id]) for p_id in path]
                                cm_value = row[cm_col]
                                probable_relationships = get_relationships_by_cm(cm_value)
                                mermaid_data = generate_mermaid_graph(path, root_id, pid, common_ancestor)
                                results_list.append({
                                    'match_name': gedcom_name,
                                    'cm': cm_value,
                                    'text_path': " → ".join(nomes),
                                    'mermaid_data': mermaid_data,
                                    'relationships': ", ".join(probable_relationships)
                                })
                            break
                results_list_sorted = sorted(results_list, key=lambda x: x.get('cm', 0), reverse=True)
                return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, dna_results=results_list_sorted, message=f"{len(results_list_sorted)} conexões encontradas.", success=True)
            except Exception as e:
                return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Ocorreu um erro: {e}", success=False)

        if action == "path_search":
            try:
                person1_name, person2_name = request.form["person1_name"], request.form["person2_name"]
                p1_ids, p2_ids = find_person_by_name(person1_name), find_person_by_name(person2_name)
                if not p1_ids:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Pessoa 1 '{person1_name}' não encontrada.", success=False)
                if not p2_ids:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Pessoa 2 '{person2_name}' não encontrada.", success=False)
                p1_id, p2_id = p1_ids[0], p2_ids[0]
                path, common_ancestor = find_ancestral_path(p1_id, p2_id)
                if not path:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Nenhuma conexão encontrada entre '{person1_name}' e '{person2_name}'.", success=True)
                nomes = [get_name(people[p_id]) for p_id in path]
                mermaid_data = generate_mermaid_graph(path, p1_id, p2_id, common_ancestor)
                path_result = {
                    'person1_name': person1_name,
                    'person2_name': person2_name,
                    'text_path': " → ".join(nomes),
                    'mermaid_data': mermaid_data
                }
                return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, path_result=path_result, message="Conexão encontrada!", success=True)
            except Exception as e:
                return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Ocorreu um erro: {e}", success=False)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
