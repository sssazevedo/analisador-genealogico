import os
import re
from flask import Flask, render_template, request
from ged4py.parser import GedcomReader
from thefuzz import fuzz
import pandas as pd
import unicodedata, string
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
child_to_family = {}  # índice filho -> [famílias onde é CHIL]

# --- Faixas cM (ampliadas) ---
SHARED_CM_DATA = [
    {"range": (3300, 3720), "relationship": "Pai/Mãe ↔ Filho(a)"},
    {"range": (2200, 3400), "relationship": "Irmãos completos"},
    {"range": (1317, 2312), "relationship": "Avós/Netos, Tios/Tias ↔ Sobrinhos(as), Meios-irmãos"},
    {"range": (553, 1330), "relationship": "Primos de 1º grau"},
    {"range": (200, 850), "relationship": "Primos de 1º grau (1× removido), Meios-primos, Tios-avós ↔ Sobrinhos-netos"},
    {"range": (46, 515), "relationship": "Primos de 2º grau"},
    {"range": (30, 350), "relationship": "Primos de 2º grau (1× removido), Primos de 3º grau"},
    {"range": (10, 220), "relationship": "Primos de 3º grau (1× removido), Primos de 4º grau"},
    {"range": (0, 110), "relationship": "Primos de 4º/5º grau ou mais distantes"},
]

# ---------- Helpers ----------
def ref_id(val):
    return getattr(val, 'xref_id', val)

def get_name(person):
    return person.name.format() if person and person.name else "Sem Nome"

STOP_WORDS = {"de","da","do","das","dos","e"}
GENERIC_GIVENS = {"maria","jose","josé","joao","joão","ana","luiz","luís","francisco",
                  "antonio","antônio","fernando","carlos","paulo","pedro","marcos", 
                  "augusto", "sergio", "sérgio", "helena"}
SURNAME_SUFFIXES = {"filho", "neto", "junior", "júnior", "sobrinho"}
COMMON_SURNAMES = {
    "silva","oliveira","santos","souza","souza","pereira","ferreira","almeida",
    "costa","rodrigues","lima","gomes","ribeiro","carvalho","azevedo","albuquerque",
}

SURNAME_EQUIV = {
    "netto": "neto",
    "gouvea": "gouveia",
    "gouvêa": "gouveia",
    "gouvéia": "gouveia",
}

def strip_bad_utf(s: str) -> str:
    if s is None: return ""
    s = str(s)

    # Correções mais comuns de mojibake (Latin-1 lido como UTF-8)
    fixes = {
        "A�": "ç", "Ã§": "ç",
        "Ã£": "ã", "Ã¡": "á", "Ã¢": "â",
        "Ã©": "é", "Ãª": "ê", "Ã¨": "è",
        "Ã­": "í", "Ã³": "ó", "Ã´": "ô", "Ãº": "ú",
        "Ã": "Ã",  
        "GouvA�": "Gouvê",   
        "A�": "ç",           
        "JoA�o": "João", "SA�": "Sá", "GonA�": "Gonç",
    }
    for bad, good in fixes.items():
        s = s.replace(bad, good)

    # Remove demais lixos não alfanuméricos (mantém letras acentuadas e ' )
    return re.sub(r"[^\w\sÁ-ú'-]", " ", s)

def norm_name(s: str) -> str:
    s = strip_bad_utf(str(s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("/", " ")
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = " ".join(s.lower().split())
    return s

SHORT_KEEP = {"sa", "sá"}  # sobrenome real na sua base

def drop_short_tokens(S, n=3):
    return {t for t in S if len(t) >= n or t in SHORT_KEEP}

def surname_core_tokens(name: str, keep_last=3):
    n = norm_name(name)
    toks = [t for t in n.split() if t and t not in STOP_WORDS]
    suffixes = []
    while toks and toks[-1] in SURNAME_SUFFIXES:
        suffixes.append(toks.pop())
    base = [t for t in toks[-keep_last:] if t not in GENERIC_GIVENS]
    # NOVO: normaliza grafias equivalentes
    base = [SURNAME_EQUIV.get(t, t) for t in base]
    return base, suffixes

def split_name_pt(s: str):
    n = norm_name(s)
    toks = [t for t in n.split() if t]
    given = next((t for t in toks if t not in STOP_WORDS and t not in GENERIC_GIVENS), toks[0] if toks else "")
    base, suffixes = surname_core_tokens(s, keep_last=3)
    surnames = [t for t in base if t != given and t not in GENERIC_GIVENS]
    return given, surnames, set(suffixes)

def surnames_set(name: str):
    _, surnames, _ = split_name_pt(name)
    return set(surnames)


def top_given_tokens(name: str, k=2):
    """retorna até k primeiros tokens 'não-stop' para comparar como possíveis 'given'."""
    n = norm_name(name)
    toks = [t for t in n.split() if t not in STOP_WORDS]
    return toks[:k] or n.split()[:k]


def token_prefixes(tokens, min_len=3):
    """gera prefixos (para lidar com 'Oliv', 'Azev', 'GouvA�..')"""
    out = set()
    for t in tokens:
        t = norm_name(t)
        if len(t) >= min_len:
            out.add(t[:min_len])
    return out

def demojibake(s: str) -> str:
    if not s:
        return s
    # só tenta se aparecerem padrões típicos
    if any(p in s for p in ("Ã", "Â", "A�", "�")):
        try:
            fixed = s.encode("latin1").decode("utf-8")
            # mantém apenas se “melhorou”
            if "�" not in fixed and "Ã" not in fixed and "A�" not in fixed:
                return fixed
        except Exception:
            pass
    return s

def soft_prefix_jaccard(a: set[str], b: set[str], min_pref=4, min_len=2) -> float:
    # filtra tokens curtíssimos (iniciais)
    a = {t for t in a if len(t) >= min_len}
    b = {t for t in b if len(t) >= min_len}
    if not a and not b:
        return 0.0
    a_short = any(len(t) <= min_pref or t.endswith(".") for t in a)
    b_short = any(len(t) <= min_pref or t.endswith(".") for t in b)
    if a_short or b_short:
        def prefset(S): return {t if len(t) <= min_pref else t[:min_pref] for t in S}
        ap, bp = prefset(a), prefset(b)
        inter = len(ap & bp); union = len(ap | bp) or 1
        return inter / union
    inter = len(a & b); union = len(a | b) or 1
    return inter / union



def get_relationships_by_cm(cm_value):
    if not isinstance(cm_value, (int, float)) or cm_value <= 0:
        return []
    poss = [item["relationship"] for item in SHARED_CM_DATA if item["range"][0] <= cm_value <= item["range"][1]]
    return poss if poss else ["Relação distante ou indeterminada"]

def load_gedcom_and_build_graph(file_path):
    global people, families, graph, child_to_family
    with GedcomReader(file_path) as parser:
        people   = {ref_id(i.xref_id): i for i in parser.records0("INDI")}
        families = {ref_id(f.xref_id): f for f in parser.records0("FAM")}
        graph, child_to_family = build_graph_from_parser(people, parser)
        all_names = sorted([get_name(p) for p in people.values()])
        return all_names

def build_graph_from_parser(people_dict, parser):
    g = nx.Graph()
    c2f = {}
    for pid, person in people_dict.items():
        g.add_node(pid, label=get_name(person), type='person')
    for fam in parser.records0("FAM"):
        fam_id = ref_id(fam.xref_id)
        if not fam_id: continue
        g.add_node(fam_id, label='Familia', type='family')
        husband_id, wife_id = None, None
        child_ids = []
        for sub_rec in fam.sub_records:
            if sub_rec.tag == "HUSB": husband_id = ref_id(sub_rec.value)
            elif sub_rec.tag == "WIFE": wife_id = ref_id(sub_rec.value)
            elif sub_rec.tag == "CHIL":
                cid = ref_id(sub_rec.value)
                child_ids.append(cid)
                c2f.setdefault(cid, []).append(fam_id)
        if husband_id and g.has_node(husband_id): g.add_edge(husband_id, fam_id)
        if wife_id and g.has_node(wife_id): g.add_edge(wife_id, fam_id)
        for cid in child_ids:
            if g.has_node(cid): g.add_edge(cid, fam_id)
    return g, c2f

def find_person_by_name(name_query):
    exact = [pid for pid, p in people.items() if name_query.lower() == get_name(p).lower()]
    if exact: return exact
    return [pid for pid, p in people.items() if name_query.lower() in get_name(p).lower()]

def get_parents(person_id):
    person = people.get(person_id)
    if not person: return []
    famc_ref = next((ref_id(rec.value) for rec in person.sub_records if rec.tag == "FAMC"), None)
    fam_ids = [famc_ref] if famc_ref else child_to_family.get(person_id, [])
    if not fam_ids: return []
    parent_ids = []
    for fam_id in fam_ids:
        family = families.get(fam_id)
        if not family: continue
        for sub_rec in family.sub_records:
            if sub_rec.tag in ("HUSB", "WIFE"):
                pid = ref_id(sub_rec.value)
                if pid and pid not in parent_ids:
                    parent_ids.append(pid)
    return parent_ids

def get_spouses(person_id):
    person = people.get(person_id)
    spouse_ids = []
    if person:
        fams_refs = [ref_id(rec.value) for rec in person.sub_records if rec.tag == "FAMS"]
        for fam_ref in fams_refs:
            family = families.get(fam_ref)
            if not family: continue
            is_husband = any(ref_id(rec.value) == person_id for rec in family.sub_records if rec.tag == "HUSB")
            partner_tag = "WIFE" if is_husband else "HUSB"
            for rec in family.sub_records:
                if rec.tag == partner_tag:
                    spouse_ids.append(ref_id(rec.value))
    if spouse_ids: return spouse_ids
    for fam in families.values():
        husb = next((ref_id(r.value) for r in fam.sub_records if r.tag == "HUSB"), None)
        wife = next((ref_id(r.value) for r in fam.sub_records if r.tag == "WIFE"), None)
        if husb == person_id and wife: spouse_ids.append(wife)
        elif wife == person_id and husb: spouse_ids.append(husb)
    return spouse_ids

def find_ancestral_path(start_id, end_id, max_depth=20):
    q1, q2 = deque([(start_id, [start_id])]), deque([(end_id, [end_id])])
    visited1, visited2 = {start_id: [start_id]}, {end_id: [end_id]}
    if start_id == end_id: return ([start_id], start_id)
    for _depth in range(max_depth):
        q_size = len(q1)
        if not q_size: break
        for _ in range(q_size):
            curr_id, path = q1.popleft()
            if curr_id in visited2: return (path + visited2[curr_id][::-1][1:], curr_id)
            for p_id in get_parents(curr_id):
                if p_id not in visited1:
                    new_path = path + [p_id]; visited1[p_id] = new_path; q1.append((p_id, new_path))
        q_size = len(q2)
        if not q_size: break
        for _ in range(q_size):
            curr_id, path = q2.popleft()
            if curr_id in visited1: return (visited1[curr_id] + path[::-1][1:], curr_id)
            for p_id in get_parents(curr_id):
                if p_id not in visited2:
                    new_path = path + [p_id]; visited2[p_id] = new_path; q2.append((p_id, new_path))
    return (None, None)

def generate_mermaid_graph(path, p1_id, p2_id, common_ancestor_id):
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
        s = (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
        s = s.replace('"', "'")
        s = re.sub(r'[\r\n]+', ' ', s)
        return s

    lines = ["flowchart BT"]; seen_nodes = set()
    ac_idx = -1
    if common_ancestor_id and common_ancestor_id in path:
        try: ac_idx = path.index(common_ancestor_id)
        except ValueError: pass
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
        if node_id in couple_members_to_skip: continue
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
                
                root_person_ids = find_person_by_name(root_name)
                if not root_person_ids:
                    return render_template("index.html", gedcom_filename=gedcom_filename, all_names=all_names, message=f"Seu nome '{root_name}' não foi encontrado no GEDCOM.", success=False)
                root_id = root_person_ids[0]

                # leitura tolerante
                try:
                    dna_matches_df = pd.read_csv(matches_path, encoding='utf-8', skipinitialspace=True)
                except UnicodeDecodeError:
                    dna_matches_df = pd.read_csv(matches_path, encoding='latin-1', skipinitialspace=True)
                
                dna_matches_df.columns = [col.strip() for col in dna_matches_df.columns]
                
                # depois de dna_matches_df.columns = [...]
                name_col = next((col for col in ['Name','MatchedName','Nome'] if col in dna_matches_df.columns), None)
                cm_col   = next((col for col in ['cM','TotalCM','Total cM'] if col in dna_matches_df.columns), None)

                # tente achar uma coluna de ID de match (p.ex., "ZH7115669", "NB8637176"...)
                id_cols_guess = [c for c in dna_matches_df.columns
                                 if dna_matches_df[c].astype(str).str.fullmatch(r'[A-Z]{2}\d{7}').mean() > 0.3]
                match_id_col = id_cols_guess[0] if id_cols_guess else None

                email_cols = [c for c in dna_matches_df.columns if 'mail' in c.lower()]
                match_email_col = email_cols[-1] if email_cols else None  # costuma ser o e-mail do parente

                def build_group_key(row):
                    parts = [norm_name(demojibake(str(row[name_col])))]
                    if match_id_col:
                        parts.append(str(row[match_id_col]).strip().upper())
                    elif match_email_col:
                        parts.append(norm_name(str(row[match_email_col])))
                    return " | ".join(parts)

                dna_matches_df["_group_key"] = dna_matches_df.apply(build_group_key, axis=1)

                aggregated_matches = (
                    dna_matches_df
                    .groupby("_group_key", as_index=False)
                    .agg({cm_col: "sum"})
                    .merge(
                        dna_matches_df[["_group_key", name_col]].drop_duplicates("_group_key"),
                        on="_group_key", how="left"
                    )
                )


                # Índices
                ged_index = {}
                surname_index = {}
                given_index = {}

                for pid, person in people.items():
                    nm = get_name(person)
                    key = norm_name(nm)
                    ged_index.setdefault(key, []).append(pid)
                    given, surnames, _ = split_name_pt(nm)
                    given_index.setdefault(given, []).append(pid)
                    for sn in surnames:
                        if sn:
                            surname_index.setdefault(sn, []).append(pid)

                HARD_MIN = 92
                GIVEN_MIN = 90

                results_list = []
                skipped_matches = []  # diagnóstico

                for _, row in aggregated_matches.iterrows():
                    csv_name_raw = str(row[name_col]).strip()
                    match_name = demojibake(csv_name_raw)
                    key = norm_name(match_name)

                    cm_value = row[cm_col]

                    # 1) exact match normalizado
                    candidate_pids = ged_index.get(key, [])

                    reason = None

                    # 2) restringe por sobrenome(s)
                    if not candidate_pids:
                        given_csv, surn_csv, csv_suffixes = split_name_pt(match_name)
                        given_norm = norm_name(given_csv)
                        # set normalizado dos sobrenomes do CSV (já com limpeza/acentos)
                        csv_surn_all = drop_short_tokens(surnames_set(match_name))

                        # monte o pool por sobrenome
                        pool = set()
                        for sn in surn_csv:
                            pool.update(surname_index.get(sn, []))

                        # fallback controlado para abreviações (ex.: 'oliv' ~ 'oliveira', 'azev' ~ 'azevedo')
                        if not pool and surn_csv:
                            pref = token_prefixes(surn_csv, min_len=3)
                            for sn, pids in surname_index.items():
                                if any(sn.startswith(p) for p in pref):
                                    pool.update(pids)

                        if not pool:
                            reason = "sem candidatos por sobrenome (abreviação/corrupção?)"
                        else:
                            # 2c) escolha do melhor candidato (por score + interseção)
                            best_pid = None
                            best_score = best_g = -1
                            best_inter = -1

                            key_norm = norm_name(match_name)

                            for pid in pool:
                                nm = get_name(people[pid])

                                # given: compare com até 2 primeiros tokens “não-stop” do GED
                                ged_given_candidates = [norm_name(t) for t in top_given_tokens(nm, k=2)]
                                s_given = max((fuzz.ratio(given_norm, gg) for gg in ged_given_candidates), default=0)

                                s_token = fuzz.token_sort_ratio(key_norm, norm_name(nm))
                                s_part  = fuzz.partial_ratio(key_norm, norm_name(nm))

                                ged_surn_set = surnames_set(nm)
                                inter_set = csv_surn_all & ged_surn_set
                                inter_cnt_local = len(inter_set)

                                # penalização para interseção apenas com sobrenomes super comuns
                                common_penalty = sum(1 for s in inter_set if s in COMMON_SURNAMES)
                                inter_bonus = 8.0 * inter_cnt_local - 4.0 * common_penalty

                                score = round(0.55*s_token + 0.25*s_part + 0.20*s_given + inter_bonus, 2)

                                # desempate: mais interseção > maior given > maior score
                                if (inter_cnt_local > best_inter or
                                    (inter_cnt_local == best_inter and s_given > best_g) or
                                    (inter_cnt_local == best_inter and s_given == best_g and score > best_score)):
                                    best_pid, best_score, best_g, best_inter = pid, score, s_given, inter_cnt_local

                            # 2d) regras de ACEITAÇÃO usando apenas o melhor candidato
                            if best_pid is not None:
                                nm_best = get_name(people[best_pid])
                                ged_surn_best = surnames_set(nm_best)
                                inter_best = csv_surn_all & ged_surn_best
                                inter_cnt = len(inter_best)

                                # se ambos têm sobrenomes e não há nenhum em comum, bloquear (a menos de sufixo ‘salvador’)
                                ged_tokens = set(norm_name(nm_best).split())
                                suffix_hit = bool(csv_suffixes & ged_tokens)
                                if csv_surn_all and ged_surn_best and inter_cnt == 0 and not suffix_hit:
                                    candidate_pids = []
                                    reason = "sem sobrenome em comum (filtro anti-falso-positivo)"
                                else:
                                    # interseção mínima adaptativa
                                    required_intersection = 1
                                    if given_norm in GENERIC_GIVENS and len(csv_surn_all) >= 2:
                                        required_intersection = 2

                                    # Jaccard com prefixo quando CSV está abreviado
                                    jacc = soft_prefix_jaccard(csv_surn_all, ged_surn_best, min_pref=4)
                                    jacc_ok = True
                                    if len(csv_surn_all) >= 2:
                                        threshold = 0.5
                                        # leve relaxamento para matches muito altos
                                        if float(cm_value or 0) >= 150 and given_norm not in GENERIC_GIVENS:
                                            threshold = 0.33
                                        jacc_ok = jacc >= threshold

                                    ACCEPT = False

                                    # (A) Dado GENÉRICO, mas sobrenomes muito fortes
                                    # Ex.: "João Batista Resende Sá": inter>=2, jacc alto, score alto
                                    if (given_norm in GENERIC_GIVENS and inter_cnt >= 2 and jacc >= 0.67 and best_score >= 100):
                                        ACCEPT = True

                                    # (B) Dado NÃO-GENÉRICO + 2 sobrenomes fortes
                                    # Ex.: "Neide Oliveira Azevedo": dado não genérico, inter>=2, jacc>=0.5
                                    elif (given_norm not in GENERIC_GIVENS and inter_cnt >= 2 and jacc >= 0.50 and best_g >= 85 and best_score >= 80):
                                        ACCEPT = True

                                    # (C) Dado NÃO-GENÉRICO + 1 sobrenome mas Jaccard MUITO alto (inicial no meio, ex.: "S")
                                    # Ex.: "Hermano S Albuquerque": inter>=1, jacc>=0.80, score razoável
                                    elif (given_norm not in GENERIC_GIVENS and inter_cnt >= 1 and jacc >= 0.80 and best_score >= 86):
                                        ACCEPT = True

                                    # (D) Regras originais
                                    elif (best_g >= 90 and best_score >= 92 and inter_cnt >= required_intersection and jacc_ok):
                                        ACCEPT = True
                                    elif (best_g >= 95 and best_score >= 88 and inter_cnt >= required_intersection and jacc_ok):
                                        ACCEPT = True
                                    else:
                                        # fallback por prefixo (mantém seu bloco atual):
                                        pref_csv = token_prefixes(surn_csv, min_len=3)
                                        if pref_csv:
                                            cand_given, cand_surns, _ = split_name_pt(nm_best)
                                            if any(sn.startswith(tuple(pref_csv)) for sn in cand_surns) and best_g >= 90 and best_score >= 86 and inter_cnt >= 1 and jacc_ok:
                                                ACCEPT = True

                                    # endurecimento por "middle" distintivo ausente (mantenha seu bloco existente)
                                    if ACCEPT:
                                        csv_tokens = drop_short_tokens({t for t in norm_name(match_name).split() if t not in STOP_WORDS})
                                        csv_middle = csv_tokens - {given_norm} - csv_surn_all
                                        if csv_middle and csv_middle.isdisjoint(ged_tokens):
                                            ACCEPT = (best_score >= 96 and best_g >= 92 and inter_cnt >= required_intersection and jacc_ok)


                                    candidate_pids = [best_pid] if ACCEPT else []
                                    if not candidate_pids:
                                        reason = (f"score insuficiente ou conflito de sobrenome "
                                                  f"(given={best_g}, final={best_score}, inter={inter_cnt}/{required_intersection}, jacc={jacc:.2f})")


                    if not candidate_pids:
                        skipped_matches.append({"csv_name": csv_name_raw, "motivo": reason or "não encontrado"})
                        continue

                    # 3) tenta achar caminho; usa o primeiro que tem caminho
                    added = False
                    for pid in candidate_pids:
                        path, common_ancestor = find_ancestral_path(root_id, pid)
                        if path:
                            nomes = [get_name(people[p_id]) for p_id in path]
                            probable_relationships = get_relationships_by_cm(cm_value)
                            mermaid_data = generate_mermaid_graph(path, root_id, pid, common_ancestor)
                            results_list.append({
                                'match_name': get_name(people[pid]),
                                'cm': cm_value,
                                'text_path': " → ".join(nomes),
                                'mermaid_data': mermaid_data,
                                'relationships': ", ".join(probable_relationships),
                                'csv_name': csv_name_raw
                            })
                            added = True
                            break
                    if not added:
                        skipped_matches.append({"csv_name": csv_name_raw, "motivo": "sem caminho subindo por pais (pais ausentes no GED?)"})

                results_list_sorted = sorted(results_list, key=lambda x: x.get('cm', 0), reverse=True)
                # Passa também os descartados para auditoria
                return render_template(
                    "index.html",
                    gedcom_filename=gedcom_filename,
                    all_names=all_names,
                    dna_results=results_list_sorted,
                    skipped_matches=skipped_matches,
                    message=f"{len(results_list_sorted)} conexões encontradas. {len(skipped_matches)} descartadas.",
                    success=True
                )
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
