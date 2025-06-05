
def all_fullkv_agent(state=None):
    return "fullkv"

def all_snapkv_64_agent(state=None):
    return "snapkv_64"

def rule_based_by_price(state=None):
    if state["energy_price"] <= 0:
        return "fullkv"
    else:
        return "snapkv_64"