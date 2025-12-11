import xml.etree.ElementTree as ET
from pathlib import Path
import csv

DESC_PATH = Path("data/desc2025.xml")
OUT_DISEASES = Path("data/mesh_diseases.tsv")
OUT_SYMPTOMS = Path("data/mesh_symptoms.tsv")

if not DESC_PATH.exists():
    raise FileNotFoundError(f"Missing {DESC_PATH}")

# Tree number prefixes
DISEASE_PREFIX = "C"            # MeSH Diseases
SYMPTOM_PREFIX = "C23.888"      # Signs and Symptoms branch


def iter_descriptors(xml_path):
    """Stream over DescriptorRecord entries to avoid high memory."""
    context = ET.iterparse(xml_path, events=("end",))
    for event, elem in context:
        if elem.tag == "DescriptorRecord":
            ui = elem.findtext("DescriptorUI")
            name = elem.findtext("DescriptorName/String")
            trees = [tn.text for tn in elem.findall("TreeNumberList/TreeNumber")]
            yield ui, name, trees
            elem.clear()


def classify_descriptors():
    diseases = []
    symptoms = []

    for ui, name, trees in iter_descriptors(DESC_PATH):
        if not trees:
            continue
        is_disease = any(t.startswith(DISEASE_PREFIX) for t in trees)
        is_symptom = any(t.startswith(SYMPTOM_PREFIX) for t in trees)

        # Signs/symptoms branch is handled separately
        if is_symptom:
            symptoms.append((ui, name, ";".join(trees)))
            continue
        if is_disease:
            diseases.append((ui, name, ";".join(trees)))

    return diseases, symptoms


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["mesh_ui", "name", "tree_numbers"])
        w.writerows(rows)


def main():
    print("Parsing desc2025.xml ...")
    diseases, symptoms = classify_descriptors()
    print(f"Diseases: {len(diseases)} | Symptoms: {len(symptoms)}")

    print(f"Writing {OUT_DISEASES}")
    write_tsv(OUT_DISEASES, diseases)
    print(f"Writing {OUT_SYMPTOMS}")
    write_tsv(OUT_SYMPTOMS, symptoms)
    print("Done.")


if __name__ == "__main__":
    main()
