import xml.dom.minidom as dom
import networkx as nx

class KmlReader:

	def __init__(self, path_):
		self.path = path_
		self.Placemarks = []
		self.ppoints = []
		self.pstrings = []
		self.nodes = {}
		self.links = []
		self.dom_ = "None"

	def create_dom(self):
		self.dom_ = dom.parse('SRIB.kml')

	def find_Placemarks(self):
		self.create_dom()
		self.Placemarks = self.dom_.getElementsByTagName('Placemark')

	def sep_points_links(self):
		for Placemark in self.Placemarks:
			points = Placemark.getElementsByTagName('Point')
			if len(points) > 0:
				self.ppoints.append(Placemark)
			else:
				self.pstrings.append(Placemark)

	def find_points(self):
		for point in self.ppoints:
			namelist = point.getElementsByTagName('name')
			nameobj = namelist[0]
			name = str(nameobj.firstChild.data)
			pointlist = point.getElementsByTagName('Point')
			coorlist = pointlist[0].getElementsByTagName('coordinates')
			coor = str(coorlist[0].firstChild.data)
			coordinates = [float(co) for co in coor.split(",")]
			self.nodes[name] = tuple(coordinates)

	def find_links(self):
		for link in self.pstrings:
			namelist = link.getElementsByTagName('name')
			names = str(namelist[0].firstChild.data)
			name = [num for num in names.split("-")]
			name = tuple(name)
			self.links.append(name)

	def load_kml(self):
		self.create_dom()
		self.find_Placemarks()
		self.sep_points_links()
		self.find_points()
		self.find_links()

	def get_nodes(self):
		return self.nodes

	def get_links(self):
		return self.links

class CreateGraph:

	def __init__(self):
		self.nodes = None
		self.links = None
		self.graph = None

	def graph_from_file(self, _path):
		kml = KmlReader(_path)
		kml.load_kml()
		self.nodes = kml.get_nodes()
		self.links = kml.get_links()

	def create_graph(self):
		self.graph = nx.DiGraph()
		self.graph.add_edges_from(self.links)
		for node in self.nodes:
			self.graph.add_node(node,coordinates = self.nodes[node])

	def check_links(self,one,two):
		links1 = self.graph.neighbors(one)
		if two in links1:
			return True
		return False

	def check_second_link(self,one,three):
		links2 = self.graph.neighbors(one)
		for link in links2:
			if self.check_links(link,three) == True:
				return True 
		return False

	def check_order(self,one,two):
		if self.check_links(one,two) == True:
			return True
		if self.check_second_link(one,two) == True:
			return True
		else:
			return False
