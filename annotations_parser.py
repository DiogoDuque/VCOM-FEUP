from xml.dom import minidom
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

def getTextFromTag(xmldoc, tagname):
    elem = xmldoc.getElementsByTagName(tagname)[0]
    rc = []
    for node in elem.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def get_filenames(dirs):
    filenames = []
    for dir in dirs:
        dir_path = os.path.join("annotations", dir)
        for filename in os.listdir(dir_path):
            if filename.endswith(".xml"):
                filenames.append(os.path.join(dir_path, filename))
    return filenames

def parse_infos(files_infos):
    results = []
    for filename in files_infos:
        xmldoc = minidom.parse(filename)
        a = Annotation()
        a.filename = getTextFromTag(xmldoc, "filename")
        a.width = getTextFromTag(xmldoc, "width")
        a.height = getTextFromTag(xmldoc, "height")
        a.depth = getTextFromTag(xmldoc, "depth")
        a.pose = getTextFromTag(xmldoc, "pose")
        a.xmin = getTextFromTag(xmldoc, "xmin")
        a.xmax = getTextFromTag(xmldoc, "xmax")
        a.ymin = getTextFromTag(xmldoc, "ymin")
        a.ymax = getTextFromTag(xmldoc, "ymax")
        results.append(a)
    return results
  
# main stuff, returns a list of Annotation
annotations = parse_infos(get_filenames(classes))

# can be deleted, it's just for displaying the values
for a in annotations:
    print(a.toString())