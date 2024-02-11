from rdkit import Chem
from rdkit.Chem import Descriptors

def lipinski_report(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "Invalid molecular structure"

    violations = []
    
    if Descriptors.NumHDonors(mol) > 5:
        violations.append("HBD > 5")

    if Descriptors.NumHAcceptors(mol) > 10:
        violations.append("HBA > 10")

    if Descriptors.MolWt(mol) > 500:
        violations.append("MW > 500 Da")

    if Descriptors.MolLogP(mol) > 5:
        violations.append("LogP > 5")
    
    if violations:
        return str("Failed: " + ", ".join(violations)), "red"
    return "Passed", "green"


def ghose_report(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "Invalid molecular structure"

    violations = []
    
    mw = Descriptors.MolWt(mol)
    if mw < 160 or mw > 480:
        violations.append("MW not in [160, 480] Da")

    logp = Descriptors.MolLogP(mol)
    if logp < -0.4 or logp > 5.6:
        violations.append("LogP not in [-0.4, 5.6]")

    hbd = Descriptors.NumHDonors(mol)
    if hbd > 5:
        violations.append("HBD > 5")

    hba = Descriptors.NumHAcceptors(mol)
    if hba > 10:
        violations.append("HBA > 10")

    if violations:
        return str("Failed: " + ", ".join(violations)), "red"
    return "Passed", "green"


def veber_report(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ""

    violations = []

    rot_bonds = Descriptors.NumRotatableBonds(mol)
    if rot_bonds > 10:
        violations.append("RotBonds>10")

    tpsa = Descriptors.TPSA(mol)
    if tpsa > 140:
        violations.append("TPSA>140")

    if violations:
        return str("Failed: " + ", ".join(violations)), "red"
    return "Passed", "green"

def egan_report(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ""

    violations = []

    tpsa = Descriptors.TPSA(mol)
    if tpsa >= 131.6:
        violations.append("TPSA >= 150")

    logp = Descriptors.MolLogP(mol)
    if logp >= 5.88:
        violations.append("LogP > 5.6")

    if violations:
        return str("Failed: " + ", ".join(violations)), "red"
    return "Passed", "green"



def muegge_report(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ""

    violations = []

    mw = Descriptors.MolWt(mol)
    xlogp = Descriptors.MolLogP(mol)
    topsa = Descriptors.TPSA(mol)
    num_rings = len(Chem.GetSSSR(mol))
    num_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    num_heteroatoms = mol.GetNumAtoms() - num_carbon
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)

    if not (200 <= mw <= 600):
        violations.append("MW not in [200, 600] Da")

    if not (-2 <= xlogp <= 5):
        violations.append("XLogP not in [-2, 5]")

    if topsa > 150:
        violations.append("TPSA > 150")

    if num_rings > 7:
        violations.append("Number of rings > 7")

    if num_carbon < 4:
        violations.append("Number of carbon atoms < 4")

    if num_heteroatoms <= 1:
        violations.append("Number of heteroatoms <= 1")

    if num_rotatable_bonds > 15:
        violations.append("Number of rotatable bonds > 15")

    if hba > 10:
        violations.append("HBA > 10")

    if hbd > 5:
        violations.append("HBD > 5")

    if violations:
        return str("Failed: " + ", ".join(violations)), "red"
    return "Passed", "green"

def check_num_violations(smiles):
    num_of_violations = sum("Failed" in report(smiles)[0] for report in [lipinski_report, egan_report, muegge_report, ghose_report, veber_report])
    if num_of_violations == 1:
        return "Only 1 violation: Molecule is likely drug-like", "green" 
    if num_of_violations == 2:
        return "Only 2 violations: Molecules may be drug-like", "green"
    if num_of_violations == 0:
        return "0 violations: Molecule is  Drug-Like", "green"
    if num_of_violations > 2:
        return str(str(num_of_violations) + " violations: Molecule is NOT drug-like"), "red"