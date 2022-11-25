import xml.etree.ElementTree as ET


def process_concept(node):
    name = node[1].text
    code = node[2].text
    return (name, code)


tree = ET.parse('../data/medra/Core_MEDRT.xml')
root = tree.getroot()

concepts = []
for child in root:
    if child.tag == 'concept':
        concepts.append(process_concept(child))

