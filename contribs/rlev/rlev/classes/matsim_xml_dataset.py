import xml.etree.ElementTree as ET
import torch
import shutil
import gzip
import tempfile
from torch_geometric.data import Dataset
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from pathlib import Path
from rlev.scripts.util import setup_config
from bidict import bidict
from rlev.classes.chargers import Charger, StaticCharger, DynamicCharger
from rlev.scripts.create_population_ev import create_population_and_plans_xml_counts


class MatsimXMLDataset(Dataset):
    """
    A dataset class for parsing MATSim XML files and creating a graph
    representation using PyTorch Geometric.

    Minimal fixes:
      - Windows-safe temp dir (tempfile.gettempdir()) instead of hardcoded /tmp
      - Clean temp dir if it already exists
      - Robust XML parsing for .xml and .xml.gz
      - Fallback: if chargers XML is unreadable, set all links to 'none'
    """

    def __init__(
        self,
        config_path: Path,
        time_string: str,
        charger_list: list[Charger],
        num_agents: int = 10000,
        initial_soc: float = 0.5,
    ):
        super().__init__(transform=None)

        # Windows-safe temp root; ensure a clean sandbox
        tmp_dir = Path(tempfile.gettempdir()) / time_string
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        output_path = tmp_dir / "output"

        shutil.copytree(config_path.parent, tmp_dir)
        self.config_path = tmp_dir / config_path.name

        (
            network_file_name,
            plans_file_name,
            vehicles_file_name,
            chargers_file_name,
            counts_file_name,
        ) = setup_config(self.config_path, str(output_path))

        self.charger_xml_path = tmp_dir / chargers_file_name
        self.network_xml_path = tmp_dir / network_file_name
        self.plan_xml_path = tmp_dir / plans_file_name
        self.vehicle_xml_path = tmp_dir / vehicles_file_name
        self.counts_xml_path = tmp_dir / counts_file_name
        self.consumption_map_path = tmp_dir / "MidCarMap.csv"
        self.charger_cost = 0

        self.node_mapping: bidict[str, int] = bidict()
        self.edge_mapping: bidict[str, int] = bidict()
        self.edge_attr_mapping: bidict[str, int] = bidict()
        self.graph: Data = Data()
        self.charger_list = charger_list
        self.num_charger_types = len(self.charger_list)
        self.max_charger_cost = 0
        self.linegraph_transform = LineGraph()

        # NOTE: this preserves original behavior (truthy check). If you want
        # "use existing plans when num_agents < 0", change this to > 0.
        if num_agents:
            create_population_and_plans_xml_counts(
                self.network_xml_path,
                self.plan_xml_path,
                self.vehicle_xml_path,
                num_agents=num_agents,
                initial_soc=initial_soc,
            )

        self.create_edge_attr_mapping()
        self.parse_matsim_network()
        self.parse_charger_network_get_charger_cost()

    # --- robust XML loader (.xml and .xml.gz) ---
    def _parse_xml(self, path: Path) -> ET.ElementTree:
        p = Path(path)
        try:
            with open(p, "rb") as f:
                head2 = f.read(2)
        except FileNotFoundError:
            raise

        if head2 == b"\x1f\x8b" or str(p).endswith(".gz"):
            with gzip.open(p, "rb") as fbin:
                return ET.parse(fbin)  # bytes; ET honors XML encoding
        return ET.parse(str(p))  # let ET auto-detect declared encoding

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def _min_max_normalize(self, tensor, reverse=False):
        if reverse:
            return tensor * (self.max_mins[1] - self.max_mins[0]) + self.max_mins[0]
        return (tensor - self.max_mins[0]) / (self.max_mins[1] - self.max_mins[0])

    def create_edge_attr_mapping(self):
        self.edge_attr_mapping = {"length": 0, "freespeed": 1, "capacity": 2}
        edge_attr_idx = len(self.edge_attr_mapping)
        for charger in self.charger_list:
            self.edge_attr_mapping[charger.type] = edge_attr_idx
            edge_attr_idx += 1

    def parse_matsim_network(self):
        tree = self._parse_xml(self.network_xml_path)
        root = tree.getroot()
        matsim_node_ids = []
        node_ids = []
        node_pos = []
        edge_index = []
        edge_attr = []

        for i, node in enumerate(root.findall(".//node")):
            node_id = node.get("id")
            matsim_node_ids.append(node_id)
            node_pos.append([float(node.get("x")), float(node.get("y"))])
            self.node_mapping[node_id] = i
            node_ids.append(i)

        tot_attr = len(self.edge_attr_mapping)

        for i, link in enumerate(root.findall(".//link")):
            from_node = link.get("from")
            to_node = link.get("to")
            from_idx = self.node_mapping[from_node]
            to_idx = self.node_mapping[to_node]
            edge_index.append([from_idx, to_idx])
            curr_link_attr = torch.zeros(tot_attr)
            self.edge_mapping[link.get("id")] = i

            for key, value in self.edge_attr_mapping.items():
                if key in link.attrib:
                    if key == "length":
                        # cost upper bound for chargers
                        link_len_km = float(link.get(key)) * 0.001
                        self.max_charger_cost += max(
                            StaticCharger.price,
                            DynamicCharger.price * link_len_km,
                        )
                    curr_link_attr[value] = float(link.get(key))

            edge_attr.append(curr_link_attr)

        self.graph.x = torch.tensor(node_ids).view(-1, 1)
        self.graph.pos = torch.tensor(node_pos)
        self.graph.edge_index = torch.tensor(edge_index).t()
        self.graph.edge_attr = torch.stack(edge_attr)
        self.linegraph = self.linegraph_transform(self.graph)
        self.max_mins = torch.stack(
            [
                torch.min(self.graph.edge_attr[:, :3], dim=0).values,
                torch.max(self.graph.edge_attr[:, :3], dim=0).values,
            ]
        )
        self.graph.edge_attr[:, :3] = self._min_max_normalize(
            self.graph.edge_attr[:, :3]
        )
        self.state = self.graph.edge_attr

    def parse_charger_network_get_charger_cost(self):
        """
        Parses the charger network XML file and calculates the total charger cost.
        If the chargers file cannot be parsed, fall back to 'no chargers' so the
        environment can still run.
        """
        cost = 0

        # Try the configured chargers file
        try:
            tree = self._parse_xml(self.charger_xml_path)
        except ET.ParseError:
            # Fallback: try sibling .xml.gz if the current is plain .xml
            p = self.charger_xml_path
            gz = None
            if p.suffix == ".xml":
                gz = p.with_suffix(".xml.gz")
            if gz is not None and gz.exists():
                tree = self._parse_xml(gz)
            else:
                # Final fallback: mark every link as 'none' and return 0 cost
                # (keeps training unblocked)
                none_idx = self.edge_attr_mapping.get("none", None)
                if none_idx is not None:
                    self.graph.edge_attr[:, 3:] = 0
                    self.graph.edge_attr[:, none_idx] = 1
                self.charger_cost = 0
                return 0

        root = tree.getroot()

        # Reset charger placements
        self.graph.edge_attr[:, 3:] = torch.zeros(
            self.graph.edge_attr.shape[0], self.graph.edge_attr[:, 3:].shape[1]
        )

        for charger in root.findall(".//charger"):
            link_id = charger.get("link")
            charger_type = charger.get("type") or StaticCharger.type

            if charger_type == StaticCharger.type:
                cost += StaticCharger.price
            elif charger_type == DynamicCharger.type:
                link_idx = self.edge_mapping[link_id]
                link_attr = self.graph.edge_attr[link_idx]
                link_attr_denormalized = self._min_max_normalize(
                    link_attr[:3], reverse=True
                )
                link_len_km = (
                    link_attr_denormalized[self.edge_attr_mapping["length"]] * 0.001
                )
                cost += DynamicCharger.price * link_len_km

            self.graph.edge_attr[self.edge_mapping[link_id]][
                self.edge_attr_mapping[charger_type]
            ] = 1

        # Mark remaining links as 'none'
        tree_net = self._parse_xml(self.network_xml_path)
        root_net = tree_net.getroot()
        for link in root_net.findall(".//link"):
            link_id = link.get("id")
            default_idx = self.edge_attr_mapping.get("default", None)
            dynamic_idx = self.edge_attr_mapping.get("dynamic", None)
            none_idx = self.edge_attr_mapping.get("none", None)

            has_default = (
                default_idx is not None
                and self.graph.edge_attr[self.edge_mapping[link_id]][default_idx] == 1
            )
            has_dynamic = (
                dynamic_idx is not None
                and self.graph.edge_attr[self.edge_mapping[link_id]][dynamic_idx] == 1
            )
            if none_idx is not None and not (has_default or has_dynamic):
                self.graph.edge_attr[self.edge_mapping[link_id]][none_idx] = 1

        self.charger_cost = cost
        return cost

    def get_graph(self):
        return self.graph
