from xml.dom import minidom
import json
import os

class Annotation:
    filename = ''
    width = -1
    height = -1
    depth = -1
    #segmented
    pose = ''
    #truncated
    xmin = -1
    xmax = -1
    ymin = -1
    ymax = -1
    def toString(self):
        return ("filename: "+self.filename+", width: "+self.width+", height: "+self.height
        +", depth: "+self.depth+", pose: "+self.pose+", xmin: "+self.xmin+", xmax: "+self.xmax
        +", ymin: "+self.ymin+", ymax: "+self.ymax)

classes = ["arrabida", "camara", "clerigos", "musica", "serralves"]

#######
# parse xml functions
######

def _getTextFromXmlTag(xmldoc, tagname):
    elem = xmldoc.getElementsByTagName(tagname)[0]
    rc = []
    for node in elem.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def _getXmlFilenames(dirs):
    filenames = []
    for dir in dirs:
        dir_path = os.path.join("annotations", dir)
        for filename in os.listdir(dir_path):
            if filename.endswith(".xml"):
                filenames.append(os.path.join(dir_path, filename))
    return filenames

def _parseXmlFiles(files_infos):
    results = []
    for filename in files_infos:
        xmldoc = minidom.parse(filename)
        a = Annotation()
        a.filename = _getTextFromXmlTag(xmldoc, "filename")
        a.width = _getTextFromXmlTag(xmldoc, "width")
        a.height = _getTextFromXmlTag(xmldoc, "height")
        a.depth = _getTextFromXmlTag(xmldoc, "depth")
        a.pose = _getTextFromXmlTag(xmldoc, "pose")
        a.xmin = _getTextFromXmlTag(xmldoc, "xmin")
        a.xmax = _getTextFromXmlTag(xmldoc, "xmax")
        a.ymin = _getTextFromXmlTag(xmldoc, "ymin")
        a.ymax = _getTextFromXmlTag(xmldoc, "ymax")
        results.append(a)
    return results

######
# convert json to xml
######

def _createXmlElement(xmlDoc, parent, name, text=None):
    elem = xmlDoc.createElement(name)
    if text is not None:
        if not isinstance(text, str):
            text = str(text)
        textElem = xmlDoc.createTextNode(text)
        elem.appendChild(textElem)
    parent.appendChild(elem)
    return elem

# used to create a single XML annotation
def _jsonToXml(jsonStr):
    # inits
    parsed_json = json.loads(jsonStr)
    xmlDoc = minidom.getDOMImplementation().createDocument(None, "annotation", None)
    root = xmlDoc.documentElement
    filename = parsed_json["content"].split("_")[-1]
    name = filename.split("-")[0]
    width = parsed_json["annotation"][0]["imageWidth"]
    height = parsed_json["annotation"][0]["imageHeight"]
    points = parsed_json["annotation"][0]["points"]
    xList = [points[0][0], points[1][0], points[2][0], points[3][0]]
    yList = [points[0][1], points[1][1], points[2][1], points[3][1]]
    xmin = min(xList) * width
    ymin = min(yList) * height
    xmax = max(xList) * width
    ymax = max(yList) * height


    # annotation/folder
    _createXmlElement(xmlDoc, root, "folder", "images")
    _createXmlElement(xmlDoc, root, "filename", filename)
    _createXmlElement(xmlDoc, root, "path", "images/"+filename)
    srcElem = _createXmlElement(xmlDoc, root, "source")
    _createXmlElement(xmlDoc, srcElem, "database", "Unknown")
    sizeElem = _createXmlElement(xmlDoc, root, "size")
    _createXmlElement(xmlDoc, sizeElem, "width", width)
    _createXmlElement(xmlDoc, sizeElem, "height", height)
    _createXmlElement(xmlDoc, sizeElem, "depth", 3)
    _createXmlElement(xmlDoc, root, "segmented", 0)
    objElem = _createXmlElement(xmlDoc, root, "object")
    _createXmlElement(xmlDoc, objElem, "name", name)
    _createXmlElement(xmlDoc, objElem, "pose", "Unspecified")
    _createXmlElement(xmlDoc, objElem, "truncated", 0)
    _createXmlElement(xmlDoc, objElem, "difficult", 0)
    boxElem = _createXmlElement(xmlDoc, objElem, "bndbox")
    _createXmlElement(xmlDoc, boxElem, "xmin", xmin)
    _createXmlElement(xmlDoc, boxElem, "ymin", ymin)
    _createXmlElement(xmlDoc, boxElem, "xmax", xmax)
    _createXmlElement(xmlDoc, boxElem, "ymax", ymax)
    
    return filename, xmlDoc

######
# main functions
######

def _displayAnnotationsString(annotations):
    for a in annotations:
        print(a.toString())

def getXmlFilesAnnotations():
    return _parseXmlFiles(_getXmlFilenames(classes))

def convertJsonToXmlFiles(filename):
    f = open(filename, "r")
    for line in f:
        filename, xmlDoc = _jsonToXml(line)
        f2 = open(filename.split(".")[0]+".xml", "w")
        f2.write(xmlDoc.toprettyxml())

def exampleAnnotations():
    print("===== ANOTACOES =====")
    annotations = getXmlFilesAnnotations()
    for annotation in annotations:
        xmin = annotation.xmin
        ymin = annotation.ymin
        bboxWidth = float(annotation.xmax) - float(xmin)
        bboxHeight = float(annotation.ymax) - float(ymin)
        width = annotation.width
        height = annotation.height
        filename = annotation.filename
        label = filename.split("-")[0]
        print(filename+" ("+label+"): top-left-corner=("+xmin+","+ymin+"), bboxDims=("+str(bboxWidth)+","+str(bboxHeight)+"), imgDims=("+width+","+height+")")
        print("===============")


# _displayAnnotationsString(getXmlFilesAnnotations())

def main():
    #convertJsonToXmlFiles("VCOM-annotations.json")
    exampleAnnotations()

if __name__ == "__main__":
    main()

