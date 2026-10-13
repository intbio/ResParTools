def get_heavy_atom_mapping(mol_with_H):
    """Создает словарь: индекс тяжелого атома в копии без H -> индекс в оригинале"""
    mol_no_H = Chem.RemoveHs(mol_with_H)
    mapping = {}
    
    # В RDKit нет прямого способа, но можно через сравнение атомов
    for i, atom_no_H in enumerate(mol_no_H.GetAtoms()):
        # Ищем соответствующий атом в исходной молекуле
        for atom_H in mol_with_H.GetAtoms():
            if (atom_H.GetAtomicNum() == atom_no_H.GetAtomicNum() and
                atom_H.GetDegree() == atom_no_H.GetDegree() and
                atom_H.GetFormalCharge() == atom_no_H.GetFormalCharge()):
                # Нужно более надежное сравнение через SMILES или инварианты
                mapping[i] = atom_H.GetIdx()
                break
    return mapping, mol_no_H

def match_chem_v3(mol_chem_1, mol_chem_2,
                  compare_any_bond=False,
                  match_residue_number=None,
                  match_type='heavy',  # 'heavy' или 'all'
                  timeout=30):
    """
    Находит максимальную общую подструктуру между мономером и полимером.
    
    Args:
        mol_chem_1 (Chem.Mol) - мономер
        mol_chem_2 (Chem.Mol) - полимер
        compare_any_bond (bool) - сравнивать любые связи
        match_residue_number (int) - номер остатка для фильтрации
        match_type (str) - 'heavy' (только тяжелые атомы) или 'all' (все атомы)
        timeout (int) - таймаут поиска MCS
    
    Returns:
        substructure (Chem.Mol) - подструктура с протонами
        dict_matches (dict) - соответствие индексов и имен атомов
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    
    # --- 1. Предупреждение для match_type='all' ---
    if match_type == 'all':
        print("ВНИМАНИЕ: Сопоставление по всем атомам (включая водороды)")
        print("Это может занять очень много времени для больших молекул!")
    
    # --- 2. Фильтрация по номеру остатка ---
    def filter_atoms_by_residue_number(mol, residue_number):
        """Возвращает новую молекулу, содержащую только атомы с заданным номером остатка"""
        indices_to_keep = []
        for atom in mol.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == residue_number:
                indices_to_keep.append(atom.GetIdx())
        
        if not indices_to_keep:
            print(f"Предупреждение: Остаток с номером {residue_number} не найден")
            return mol
        
        print(f"Фильтрация по остатку {residue_number}: оставлено {len(indices_to_keep)} атомов")
        return Chem.PathToSubmol(mol, indices_to_keep)
    
    if match_residue_number is not None:
        print(f'Поиск MCS только для остатка с номером {match_residue_number}')
        filtered_mol_chem_2 = filter_atoms_by_residue_number(mol_chem_2, match_residue_number)
    else:
        filtered_mol_chem_2 = mol_chem_2
    
    # --- 3. Подготовка молекул в зависимости от match_type ---
    if match_type == 'heavy':
        print("Сопоставление по тяжелым атомам (водороды исключены)")
        mol_1_processed = Chem.RemoveHs(mol_chem_1)
        mol_2_processed = Chem.RemoveHs(filtered_mol_chem_2)
    else:  # 'all'
        print("Сопоставление по всем атомам (включая водороды)")
        mol_1_processed = mol_chem_1
        mol_2_processed = filtered_mol_chem_2
    
    # --- 4. Настройка параметров MCS ---
    mcs_params = rdFMCS.MCSParams()
    mcs_params.Timeout = timeout
    mcs_params.Maximize = True
    mcs_params.CompareAnyBond = compare_any_bond
    
    if match_type == 'heavy':
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElementsAndCharge
        mcs_params.MatchValences = False
    else:
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
        mcs_params.MatchValences = True
    
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    mcs_params.RingMatchesRingOnly = False
    mcs_params.CompleteRingsOnly = False
    
    # --- 5. Поиск MCS ---
    print("Поиск максимальной общей подструктуры...")
    mcs_result = rdFMCS.FindMCS([mol_1_processed, mol_2_processed], mcs_params)
    
    if not mcs_result or mcs_result.numAtoms == 0:
        raise ValueError("Общая подструктура не найдена!")
    
    print(f"Найдена общая подструктура из {mcs_result.numAtoms} атомов")
    
    # --- 6. Получение сопоставлений ---
    query = Chem.MolFromSmarts(mcs_result.smartsString)
    matches_1 = mol_1_processed.GetSubstructMatches(query, uniquify=False)
    matches_2 = mol_2_processed.GetSubstructMatches(query, uniquify=False)
    
    # --- 7. Выбор лучшего сопоставления (максимум атомов) ---
    # best_match_1 = None
    # best_match_2 = None
    max_size = 0
    
    for match_1 in matches_1:
        for match_2 in matches_2:
            size = len(match_1)  # количество сопоставленных атомов
            if size > max_size:
                max_size = size
                best_match_1 = match_1
                best_match_2 = match_2
    
    if best_match_1 is None:
        best_match_1 = matches_1[0]
        best_match_2 = matches_2[0]
    
    print(f"Выбрано сопоставление с {max_size} атомами")
    
    # --- 8. Создание словаря соответствий ---
    dict_matches = {}
    
    # Сохраняем индексы (без сортировки!)
    dict_matches['Indexes'] = dict(zip(best_match_1, best_match_2))
    
    # Добавляем имена атомов, если они есть
    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        try:
            mol1_atoms_name = [mol_chem_1.GetAtomWithIdx(idx).GetProp('AtomName') 
                               for idx in best_match_1]
            mol2_atoms_name = [mol_chem_2.GetAtomWithIdx(idx).GetProp('AtomName') 
                               for idx in best_match_2]
            dict_matches['Names'] = dict(zip(mol1_atoms_name, mol2_atoms_name))
        except Exception as e:
            print(f"Предупреждение: Не удалось получить имена атомов - {e}")
    
    # --- 9. Построение подструктуры с протонами ---
    # Берем атомы из исходной молекулы-полимера (с водородами)
    # Используем best_match_2 как есть, без сортировки
    
    # Добавляем водороды, связанные с сопоставленными атомами
    all_indices = list(best_match_2)  # копируем список
    
    for idx in best_match_2:
        atom = mol_chem_2.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # Водород
                if neighbor.GetIdx() not in all_indices:
                    all_indices.append(neighbor.GetIdx())
    
    # Не сортируем индексы, сохраняем порядок
    substructure = Chem.PathToSubmol(mol_chem_2, all_indices)
    
    # Заряды копируются автоматически через PathToSubmol
    # Дополнительно проверяем и восстанавливаем если нужно
    for i, old_idx in enumerate(all_indices):
        old_atom = mol_chem_2.GetAtomWithIdx(old_idx)
        new_atom = substructure.GetAtomWithIdx(i)
        if old_atom.GetFormalCharge() != 0:
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
    
    return substructure, dict_matches


def rdkit_pdb_modification(rdkit_mol, resname="MOD", resid=1, segid="A",
                           force_numeric=False, C_terminal=False):
    """
    Назначает PDB имена атомов с поуровневым сбором и глобальными индексами на уровне.
    """
    from collections import deque, defaultdict
    
    greek_levels = {1: "B", 2: "G", 3: "D", 4: "E", 5: "Z", 6: "H",
                    7: "T", 8: "I", 9: "K", 10: "L", 11: "M", 12: "N"}

    atom_names = {}
    used_names = set()
    visited = set()
    numeric_mode = force_numeric
    heavy_counter = 0
    processed_hydrogens = set()
    hydrogen_counts = {}

    # ---------------------------------------------------------
    # 1. Поиск backbone
    # ---------------------------------------------------------
    CA_idx, backbone = find_backbone_match(rdkit_mol, C_terminal)

    backbone_names = ["N", "CA", "C", "O"]
    if len(backbone) == 5:
        backbone_names = ["N", "CA", "C", "OC1", "OC2"]

    for idx, name in zip(backbone, backbone_names):
        atom_names[idx] = name
        used_names.add(name)
        if rdkit_mol.GetAtomWithIdx(idx).GetSymbol() != "H":
            heavy_counter += 1
        visited.add(idx)

    N_idx = backbone[0]

    # ---------------------------------------------------------
    # 2. Назначение HA
    # ---------------------------------------------------------
    for nbr in rdkit_mol.GetAtomWithIdx(CA_idx).GetNeighbors():
        if nbr.GetSymbol() == "H" and nbr.GetIdx() not in atom_names:
            atom_names[nbr.GetIdx()] = "HA"
            used_names.add("HA")
            processed_hydrogens.add(nbr.GetIdx())

    # ---------------------------------------------------------
    # 3. Протоны аминогруппы
    # ---------------------------------------------------------
    amine_counter = 1
    for nbr in rdkit_mol.GetAtomWithIdx(N_idx).GetNeighbors():
        if nbr.GetSymbol() == "H" and nbr.GetIdx() not in atom_names:
            name = f"H{amine_counter}"
            atom_names[nbr.GetIdx()] = name
            used_names.add(name)
            processed_hydrogens.add(nbr.GetIdx())
            amine_counter += 1

    # ---------------------------------------------------------
    # 4. Поуровневый сбор атомов
    # ---------------------------------------------------------
    current_level = [(CA_idx, 0)]  # (atom_idx, depth)
    next_level = []
    
    # Словарь для сбора атомов по уровням
    # {depth: [(atom_idx, element, parent_idx)]}
    level_atoms = defaultdict(list)
    
    # BFS для сбора всех атомов по уровням
    level = 0
    while current_level:
        next_level = []
        for current_idx, depth in current_level:
            current_atom = rdkit_mol.GetAtomWithIdx(current_idx)
            
            for nbr in current_atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                
                if nbr_idx in visited or nbr_idx in backbone:
                    continue
                    
                if nbr.GetSymbol() != "H":  # Тяжелый атом
                    element = nbr.GetSymbol()
                    
                    # Сохраняем атом для этого уровня
                    level_atoms[depth + 1].append((nbr_idx, element, current_idx))
                    visited.add(nbr_idx)
                    next_level.append((nbr_idx, depth + 1))
        
        current_level = next_level
    
    # ---------------------------------------------------------
    # 5. Назначение имен по уровням
    # ---------------------------------------------------------
    for depth in sorted(level_atoms.keys()):
        atoms_on_level = level_atoms[depth]
        
        print(f"\nУровень {depth}: {len(atoms_on_level)} атомов")
        
        # Определяем букву для этого уровня
        if not numeric_mode and depth in greek_levels:
            level_letter = greek_levels[depth]
            print(f"  Буква уровня: {level_letter}")
        else:
            level_letter = None
            if not numeric_mode:
                numeric_mode = True
                print(f"  Переход в numeric режим на глубине {depth}")
        
        # Глобальный счетчик для этого уровня
        level_counter = 1
        
        # Назначаем имена ВСЕМ атомам на этом уровне
        for atom_idx, element, parent_idx in atoms_on_level:
            name = None
            
            # Greek режим
            if not numeric_mode and level_letter:
                # Базовая часть имени: элемент + буква уровня
                base = f"{element}{level_letter}"
                
                # ВСЕГДА добавляем глобальный индекс уровня
                # Даже если в имени уже есть индекс (OZ1, NZ2), мы его заменяем!
                candidate = f"{base}{level_counter}"
                
                if len(candidate) <= 4 and candidate not in used_names:
                    name = candidate
                    print(f"    {element}{level_letter} (бывший) -> {name} (индекс {level_counter})")
            
            # Numeric режим
            if name is None:
                if not numeric_mode:
                    numeric_mode = True
                    print(f"    Переход в numeric режим для {element} на глубине {depth}")
                
                heavy_counter += 1
                base_name = f"{element}{heavy_counter}"
                name = base_name[:4]
                
                # Проверка уникальности
                counter = 1
                while name in used_names:
                    name = f"{base_name}_{counter}"[:4]
                    counter += 1
                
                print(f"    {element} -> {name} (numeric)")
            
            atom_names[atom_idx] = name
            used_names.add(name)
            
            # Сохраняем информацию о том, какой глобальный индекс получил атом
            atom_global_index = level_counter
            level_counter += 1
            
            # -------------------------------------------------
            # Водороды этого атома
            # -------------------------------------------------
            atom = rdkit_mol.GetAtomWithIdx(atom_idx)
            h_neighbors = []
            
            for h in atom.GetNeighbors():
                h_idx = h.GetIdx()
                if (h.GetSymbol() == "H" and 
                    h_idx not in processed_hydrogens and 
                    h_idx not in backbone):
                    h_neighbors.append(h)
            
            if h_neighbors:
                if name not in hydrogen_counts:
                    hydrogen_counts[name] = 1
                
                # Для водородов используем родительский индекс
                parent_index = "".join(c for c in name if c.isdigit())
                
                if numeric_mode:
                    for h in h_neighbors:
                        hname = f"H{parent_index}{hydrogen_counts[name]}"
                        hydrogen_counts[name] += 1
                        
                        if len(hname) > 4:
                            hname = f"H{parent_index}"[:4]
                        
                        # Проверка уникальности
                        counter = 1
                        base_hname = hname
                        while hname in used_names:
                            hname = f"{base_hname}_{counter}"[:4]
                            counter += 1
                        
                        atom_names[h.GetIdx()] = hname
                        used_names.add(hname)
                        processed_hydrogens.add(h.GetIdx())
                        print(f"      водород {hname}")
                else:
                    # В Greek режиме водороды наследуют родительский суффикс
                    parent_suffix = name[1:]  # все кроме первого символа
                    
                    for h in h_neighbors:
                        if len(h_neighbors) == 1:
                            hname = f"H{parent_suffix}"
                        else:
                            hname = f"H{parent_suffix}{hydrogen_counts[name]}"
                            hydrogen_counts[name] += 1
                        
                        if len(hname) > 4:
                            hname = f"H{parent_suffix}"[:4]
                        
                        # Проверка уникальности
                        counter = 1
                        base_hname = hname
                        while hname in used_names:
                            hname = f"{base_hname}_{counter}"[:4]
                            counter += 1
                        
                        atom_names[h.GetIdx()] = hname
                        used_names.add(hname)
                        processed_hydrogens.add(h.GetIdx())
                        print(f"      водород {hname}")

    # ---------------------------------------------------------
    # 6. Проверка пропущенных атомов
    # ---------------------------------------------------------
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in atom_names:
            print(f"Предупреждение: атом {idx} ({atom.GetSymbol()}) не получил имя")
            element = atom.GetSymbol()
            heavy_counter += 1
            base_name = f"{element}{heavy_counter}"
            name = base_name[:4]
            
            counter = 1
            while name in used_names:
                name = f"{base_name}_{counter}"[:4]
                counter += 1
            
            atom_names[idx] = name
            used_names.add(name)

    # ---------------------------------------------------------
    # 7. Запись PDB
    # ---------------------------------------------------------
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        name = atom_names.get(idx, "UNK")
        
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.ljust(4))
        info.SetResidueName(resname)
        info.SetResidueNumber(resid)
        info.SetChainId(segid)
        atom.SetProp("AtomName", name.strip())
        atom.SetMonomerInfo(info)

    check_duplicate_atom_names(rdkit_mol)
    return rdkit_mol


def check_duplicate_atom_names(mol):
    """
    Проверяет уникальность имён атомов в молекуле (по свойству AtomName).
    Возвращает True, если дубликатов нет.
    """
    name_to_indices = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("AtomName"):
            name = atom.GetProp("AtomName")
            idx = atom.GetIdx()
            if name not in name_to_indices:
                name_to_indices[name] = []
            name_to_indices[name].append(idx)

    duplicates = {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}
    if duplicates:
        print("⚠️ Обнаружены дубликаты имён атомов:")
        for name, indices in duplicates.items():
            print(f"   '{name}' : индексы {indices}")
        return False
    print("✅ Все имена атомов уникальны.")
    return True