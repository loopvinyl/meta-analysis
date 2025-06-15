def assign_material_group(source):
    if pd.isna(source): 
        return "Uncategorized"
    source = str(source).lower()
    
    # 1. Animal Manure-Based (Prioridade máxima)
    manure_keywords = ["manure", "dung", "cattle", "cow", "bovine", "cd", "vr", "fezes", "estrume", "gado", "vaca"]
    if any(kw in source for kw in manure_keywords):
        # Casos especiais com misturas
        if "coffee" in source or "borra" in source:
            return "Manure + Coffee Waste"
        if "pineapple" in source or "abacaxi" in source:
            return "Manure + Fruit Waste"
        if "bagasse" in source or "crop" in source:
            return "Manure + Crop Residues"
        return "Animal Manure-Based"
    
    # 2. Coffee Waste
    coffee_keywords = ["coffee", "scg", "borra", "café"]
    if any(kw in source for kw in coffee_keywords):
        return "Coffee Waste"
    
    # 3. Fruit Waste
    fruit_keywords = ["pineapple", "abacaxi", "fruit", "fruta", "peels"]
    if any(kw in source for kw in fruit_keywords):
        return "Fruit Waste"
    
    # 4. Food Waste
    if "food" in source or "kitchen" in source or "alimento" in source:
        return "Food Waste"
    
    # 5. Crop Residues
    crop_keywords = ["bagasse", "crop residue", "straw", "palha", "sugarcane", "bagaço"]
    if any(kw in source for kw in crop_keywords):
        return "Crop Residues"
    
    # 6. Green Waste
    green_keywords = ["vegetable", "grass", "water hyacinth", "weeds", "parthenium", "green", "verde"]
    if any(kw in source for kw in green_keywords):
        return "Green Waste"
    
    # 7. Cellulosic Waste
    cellulosic_keywords = ["cardboard", "paper", "filters", "filtro", "cellulose", "papel"]
    if any(kw in source for kw in cellulosic_keywords):
        return "Cellulosic Waste"
    
    return "Uncategorized"
